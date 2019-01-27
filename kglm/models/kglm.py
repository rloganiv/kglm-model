from typing import Any, Dict, List, Union

from allennlp.data import  Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from overrides import overrides
import torch
import torch.nn.functional as F

from kglm.data import AliasDatabase


StateDict = Dict[str, Union[torch.Tensor]]  # pylint: disable=invalid-name


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

        self._mention_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                        out_features=2)
        self._entity_projection = torch.nn.Linear(in_features=embedding_dim,
                                                  out_features=embedding_dim)
        self._alias_encoder = torch.nn.LSTM(input_size=embedding_dim,
                                            hidden_size=embedding_dim,
                                            batch_first=True)

        self._generate_mode_projection = torch.nn.Linear(in_features=embedding_dim,
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

        if not self.training:
            # TODO Some evaluation stuff
            pass

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
        mention_loss = sequence_cross_entropy_with_logits(logits, targets, mask)
        return mention_loss

    def _entity_prediction_loss(self,
                                hidden: torch.Tensor,
                                targets: torch.Tensor,
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
            token_losses = F.cross_entropy(logits[i], targets[i], shortlist_mask[i].float(), reduction='none')
            loss += token_losses.sum()
        loss /= batch_size

        return loss

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

        # First we predict whether or not the next token will be an entity mention.
        target_mentions = entity_mask[:, 1:].contiguous()
        mention_loss = self._mention_loss(hidden, target_mentions, target_mask)

        # Next we predict which entity (among those in the supplied shortlist) is going to be
        # mentioned.
        target_shortlist_indices = shortlist_indices[:, 1:].contiguous()
        entity_prediction_loss = self._entity_prediction_loss(hidden,
                                                              target_shortlist_indices,
                                                              shortlist_embeddings,
                                                              shortlist_mask)
        loss = mention_loss + entity_prediction_loss

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
