from typing import Any, Dict, List, Union

from allennlp.data.vocabulary import Vocabulary, DEFAULT_OOV_TOKEN
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, \
    masked_log_softmax
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data import AliasDatabase


StateDict = Dict[str, Union[torch.Tensor]]  # pylint: disable=invalid-name
LOG0 = torch.tensor(1e-34).log()


@Model.register('kglm')
class Kglm(Model):
    """
    Knowledge graph language model.

    Parameters
    ----------
    vocab : ``Vocabulary``
        The model vocabulary.
    text_field_embedder : ``TextFieldEmbedder``
        Used to embed tokens.
    encoder : ``Seq2SeqEncoder``
        Used to encode the sequence of token embeddings.
    embedding_dim : ``int``
        The dimension of entity / length embeddings. Should match the encoder output size.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 encoder: Seq2SeqEncoder,
                 token_embedder: TextFieldEmbedder,
                 entity_embedder: TextFieldEmbedder,
                 embedding_dim: int) -> None:
        super(Kglm, self).__init__(vocab)

        self._embedding_dim = embedding_dim
        self._token_embedder = token_embedder
        self._entity_embedder = entity_embedder
        self._encoder = encoder
        self._unk_index = vocab.get_token_index(DEFAULT_OOV_TOKEN)

        self._mention_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                        out_features=2)
        self._entity_projection = torch.nn.Linear(in_features=embedding_dim,
                                                  out_features=embedding_dim)
        self._alias_encoder = torch.nn.LSTM(input_size=embedding_dim,
                                            hidden_size=embedding_dim,
                                            batch_first=True)

        self._generate_mode_projection = torch.nn.Linear(in_features=2 * embedding_dim,
                                                         out_features=vocab.get_vocab_size('tokens'))
        self._copy_mode_projection = torch.nn.Linear(in_features=embedding_dim,
                                                     out_features=embedding_dim)

        self._state = None

    @overrides
    def forward(self,  # pylint: disable=arguments-differ
                tokens: Dict[str, torch.Tensor],
                reset: bool,
                metadata: List[Dict[str, Any]],
                entity_identifiers: torch.Tensor = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_indices: torch.Tensor = None,
                alias_copy_indices: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # Tensorize the alias_database - this will only perform the operation once.
        alias_database = metadata[0]['alias_database']
        alias_database.tensorize(vocab=self.vocab, device=tokens['tokens'].device)

        # Reset the model if needed
        batch_size = tokens['tokens'].shape[0]
        if reset:
            self.reset_states(batch_size)

        if entity_identifiers is not None:
            output_dict = self._forward_loop(tokens=tokens,
                                             alias_database=alias_database,
                                             entity_identifiers=entity_identifiers,
                                             shortlist=shortlist,
                                             shortlist_indices=shortlist_indices,
                                             alias_copy_indices=alias_copy_indices)

        else:
            output_dict = {}

        return output_dict

    def _mention_loss(self,
                      hidden: torch.Tensor,
                      targets: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss term for predicting whether or not the the next token will be part of
        an entity mention.
        """
        logits = self._mention_type_projection(hidden)
        mention_loss = sequence_cross_entropy_with_logits(logits, targets, mask, average="token")
        return mention_loss

    def _entity_prediction_loss(self,
                                hidden: torch.Tensor,
                                targets: torch.Tensor,
                                mask: torch.Tensor,
                                shortlist_embeddings: torch.Tensor,
                                shortlist_mask: torch.Tensor) -> torch.Tensor:
        # Logits are computed using a bilinear form that measures the similarity between the
        # projected hidden state and the embeddings of entities in the shortlist
        projected = self._entity_projection(hidden)
        logits = torch.bmm(projected, shortlist_embeddings.transpose(1, 2))

        # There are technically two masks that need to be accounted for: a class-wise mask which
        # specifies which logits to ignore in the class dimension, and a token-wise mask (e.g.
        # `mask`) which avoids measuring loss for predictions on non-mention tokens. In practice,
        # we only need the class-wise mask since the non-mention tokens cannot be associated with a
        # valid target.
        batch_size = hidden.shape[0]
        loss = 0.0
        for i in range(batch_size):
            loss += F.cross_entropy(logits[i], targets[i], shortlist_mask[i].float(), reduction='sum')
        loss /= (mask.sum().float() + 1e-13)

        return loss

    def _get_copy_scores(self,
                         hidden: torch.Tensor,
                         alias_tokens: torch.Tensor) -> torch.Tensor:
        # Begin by flattening the tokens so that they fit the expected shape of a
        # ``Seq2SeqEncoder``.
        batch_size, sequence_length, num_aliases, alias_length = alias_tokens.shape
        flattened = alias_tokens.view(-1, alias_length)
        mask = flattened == 0

        # Next we run through standard pipeline
        embedded = self._token_embedder({'tokens': flattened})  # UGLY
        encoded = self._encoder(embedded, mask)

        # Equation 8 in the CopyNet paper recommends applying the additional step.
        projected = torch.tanh(self._copy_mode_projection(encoded))

        # This part gets a little funky - we need to make sure that the first dimension in
        # `projected` and `hidden` is batch_size x sequence_length.
        hidden = hidden.view(batch_size * sequence_length, 1, -1)
        projected = projected.view(batch_size * sequence_length, -1, num_aliases * alias_length)
        copy_scores = torch.bmm(hidden, projected).squeeze()
        copy_scores = copy_scores.view(batch_size, sequence_length, -1).contiguous()

        return copy_scores

    def _get_vocab_loss(self,
                        generate_scores: torch.Tensor,
                        copy_scores: torch.Tensor,
                        target_tokens: torch.Tensor,
                        target_copy_indices: torch.Tensor,
                        mask: torch.Tensor,
                        alias_indices: torch.Tensor):

        batch_size, sequence_length, vocab_size = generate_scores.shape
        copy_vocab_size = copy_scores.shape[-1]

        # We create a mask to ensure that padding alias tokens are omitted from the softmax.
        alias_mask = alias_indices.view(batch_size, sequence_length, -1).gt(0)
        score_mask = mask.new_ones(batch_size, sequence_length, vocab_size + copy_vocab_size)
        score_mask[:, :, vocab_size:] = alias_mask

        # Next we concatenate the score tensors together in order to compute the log probabilities.
        concatenated_scores = torch.cat((generate_scores, copy_scores), dim=-1)
        log_probs = masked_log_softmax(concatenated_scores, score_mask)

        # The generated token loss is a simple cross-entropy calculation, we can just gather
        # the log probabilties...
        flattened_log_probs = log_probs.view(batch_size * sequence_length, -1)
        flattened_targets = target_tokens.view(batch_size * sequence_length, 1)
        generate_log_probs = flattened_log_probs.gather(1, flattened_targets).squeeze()
        # ...except we need to omit any <UNK> terms that correspond to copy tokens.
        unk_targets = target_tokens.eq(self._unk_index)
        copied_targets = target_copy_indices.gt(0)
        ignored_targets = unk_targets * copied_targets
        generate_mask = (1 - mask.byte()) * ignored_targets
        generate_loss = -generate_log_probs
        generate_loss[generate_mask.view(-1)] = -LOG0  # log(0)

        # The copied token loss requires adding up all of the relevant token log probabilities.
        # We'll use a for loop to keep things simple for now.
        copy_log_probs = flattened_log_probs[:, vocab_size:]
        flattened_copy_mask = copied_targets.view(-1)
        if flattened_copy_mask.sum() > 0:
            import pdb; pdb.set_trace()
        flattened_alias_indices = alias_indices.view(batch_size * sequence_length, -1)
        flattened_target_copy_indices = target_copy_indices.view(-1)
        copy_loss = torch.zeros_like(generate_loss)
        for i in range(batch_size * sequence_length):
            selection_mask = flattened_alias_indices[i].eq(flattened_target_copy_indices[i])
            if selection_mask.sum() > 0 and flattened_copy_mask[i] != 0:
                selected_log_probs = copy_log_probs[i].masked_select(selection_mask)
                total_log_prob = torch.logsumexp(selected_log_probs, dim=0)
                copy_loss[i] = -total_log_prob
            else:
                copy_loss[i] = -LOG0

        combined_loss = torch.cat((generate_loss, copy_loss), dim=-1)
        combined_loss = torch.logsumexp(combined_loss, dim=-1)
        combined_loss = combined_loss.sum() / (mask.sum().float() + 1e-13)

        return combined_loss

    def _forward_loop(self,
                      tokens: Dict[str, torch.Tensor],
                      alias_database: AliasDatabase,
                      entity_identifiers: Dict[str, torch.Tensor],
                      shortlist: Dict[str, torch.Tensor],
                      shortlist_indices: torch.Tensor,
                      alias_copy_indices: torch.Tensor) -> Dict[str, torch.Tensor]:

        batch_size, sequence_length = tokens['tokens'].shape

        # if self._state is not None:
        #     tokens = {field: torch.cat((self._state['prev_tokens'][field], tokens[field]), dim=1) for field in tokens}
        #     entity_types = torch.cat((self._state['entity_types'], entity_types), dim=1)
        #     entity_ids = {field: torch.cat((self._state['entity_ids'][field], entity_ids[field]), dim=1) for field in tokens}
        #     alias_ids = torch.cat((self._state['alias_ids'], entity_ids), dim=1)

        # Get the token mask and target tensors
        token_mask = get_text_field_mask(tokens)
        target_mask = token_mask[:, 1:].contiguous()
        target_tokens = tokens['tokens'][:, 1:].contiguous()
        target_copy_indices = alias_copy_indices[:,1:].contiguous()

        # Embed and encode the source tokens
        source_mask = token_mask[:, :-1].contiguous()
        source_embeddings = self._token_embedder(tokens)[:, :-1].contiguous()
        hidden = self._encoder(source_embeddings, source_mask)

        # Embed entities
        entity_mask = get_text_field_mask(entity_identifiers)
        entity_embeddings = self._entity_embedder(entity_identifiers)

        # Embed entity shortlist
        shortlist_mask = get_text_field_mask(shortlist)
        shortlist_embeddings = self._entity_embedder(shortlist)

        # Predict whether or not the next token will be an entity mention.
        target_mentions = entity_mask[:, 1:].contiguous()
        mention_loss = self._mention_loss(hidden, target_mentions, target_mask)

        # Predict which entity (among those in the supplied shortlist) is going to be
        # mentioned.
        target_shortlist_indices = shortlist_indices[:, 1:].contiguous()
        entity_prediction_loss = self._entity_prediction_loss(hidden,
                                                              target_shortlist_indices,
                                                              target_mask,
                                                              shortlist_embeddings,
                                                              shortlist_mask)

        # Predict generation-mode scores. Start by concatenating predicted entity embeddings with
        # the encoder output - then feed through a linear layer.
        target_embeddings = entity_embeddings[:, 1:].contiguous()
        concatenated = torch.cat((hidden, target_embeddings), dim=-1)
        generate_scores = self._generate_mode_projection(concatenated)

        # Predict copy-mode scores.
        target_entity_identifiers = entity_identifiers['entity_ids'][:, 1:].contiguous()
        alias_tokens, alias_indices = alias_database.lookup(target_entity_identifiers)
        copy_scores = self._get_copy_scores(hidden, alias_tokens)

        # Combine scores to get vocab loss
        vocab_loss = self._get_vocab_loss(generate_scores,
                                          copy_scores,
                                          target_tokens,
                                          target_copy_indices,
                                          target_mask,
                                          alias_indices)

        loss = (mention_loss + entity_prediction_loss + vocab_loss) / 3


        # for timestep in range(sequence_length - 1):

        #     current_hidden = hidden[:, timestep]
        #     current_mask = mask[:, timestep]

        #     next_entity_types = entity_types[:, timestep + 1]
        #     next_entity_ids = entity_ids['entity_ids'][:, timestep + 1]
        #     next_shortlist_ids = entity_shortlist_ids[:, timestep + 1]
        #     next_tokens = tokens['tokens'][:, timestep + 1]

        #     # Predict entity types
        #     entity_type_logits = self._entity_type_projection(current_hidden[current_mask])
        #     entity_type_loss += F.cross_entropy(entity_type_logits,
        #                                         next_entity_types.long(),
        #                                         reduction='sum')

        #     # Predict entity id (from shortlist for now)
        #     projected_hidden = self._entity_id_projection(current_hidden)
        #     projected_hidden = projected_hidden.unsqueeze(2)
        #     entity_id_logits = torch.bmm(shortlist_embeddings, projected_hidden).squeeze(2)
        #     _entity_id_loss = F.cross_entropy(entity_id_logits,
        #                                       next_shortlist_ids,
        #                                       reduction='none')
        #     _entity_id_loss = _entity_id_loss * next_entity_types.float()  # Only predict for new entities
        #     entity_id_loss += _entity_id_loss.sum()

        #     # Now for the fun part - the next word.
        #     # TODO: Finish; right now we'll just do alias scores.
        #     copy_hidden = self._copy_mode_projection(current_hidden)  # TODO: Better name - hidden proj. for copy
        #     for i in range(batch_size):
        #         # Get the global and local alias arrays
        #         if next_entity_types[i] == 0:
        #             continue
        #         import pdb; pdb.set_trace()
        #         next_entity_id = next_entity_ids[i]
        #         global_id, local_id = alias_database.lookup(next_entity_id.item())
        #         # Encode global ids
        #         global_id = global_id
        #         alias_id_mask = global_id.eq(0)
        #         global_token_embeddings = self._token_embedder({'tokens': global_id})
        #         alias_encoding, _ = self._alias_encoder(global_token_embeddings)
        #         alias_encoding = torch.tanh(alias_encoding)

        #         copy_scores = torch.einsum('ijk,k->ij', alias_encoding, copy_hidden[i, :])
        # return {}
        return {'loss': loss}

    def reset_states(self, batch_size: int) -> None:
        """Resets the model's internals. Should be called at the start of a new batch."""
        self._encoder.reset_states()
        self._state = None
