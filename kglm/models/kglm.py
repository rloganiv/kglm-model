from typing import Any, Dict, List, Union

from allennlp.data import  Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models import Model
from allennlp.nn.util import get_text_field_mask
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

        self._entity_type_projection = torch.nn.Linear(in_features=embedding_dim,
                                                       out_features=2)
        self._entity_id_projection = torch.nn.Linear(in_features=embedding_dim,
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
                entity_shortlist: Dict[str, torch.Tensor] = None,
                entity_shortlist_ids: torch.Tensor = None,
                entity_types: torch.Tensor = None,
                entity_ids: torch.Tensor = None,
                alias_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # Tensorize the alias_database - this will only perform the operation once.
        alias_database = metadata[0]['alias_database']
        alias_database.tensorize(vocab=self.vocab, device=tokens['tokens'].device)

        # Reset the model if needed
        batch_size = tokens['tokens'].shape[0]
        if reset:
            self.reset_states(batch_size)

        if entity_types is not None:
            output_dict = self._forward_loop(tokens=tokens,
                                             alias_database=alias_database,
                                             entity_shortlist=entity_shortlist,
                                             entity_shortlist_ids=entity_shortlist_ids,
                                             entity_types=entity_types,
                                             entity_ids=entity_ids,
                                             alias_ids=alias_ids)

        else:
            output_dict = {}

        if not self.training:
            # TODO Some evaluation stuff
            pass

        return output_dict

    def _forward_loop(self,
                      tokens: Dict[str, torch.Tensor],
                      alias_database: AliasDatabase,
                      entity_shortlist: Dict[str, torch.Tensor],
                      entity_shortlist_ids: torch.Tensor,
                      entity_types: torch.Tensor,
                      entity_ids: Dict[str, torch.Tensor],
                      alias_ids: torch.Tensor) -> Dict[str, torch.Tensor]:

        batch_size, sequence_length = tokens['tokens'].shape

        if self._state is not None:
            tokens = {field: torch.cat((self._state['prev_tokens'][field], tokens[field]), dim=1) for field in tokens}
            entity_types = torch.cat((self._state['entity_types'], entity_types), dim=1)
            entity_ids = {field: torch.cat((self._state['entity_ids'][field], entity_ids[field]), dim=1) for field in tokens}
            alias_ids = torch.cat((self._state['alias_ids'], entity_ids), dim=1)

        # Embed tokens
        mask = get_text_field_mask(tokens)
        embeddings = self._token_embedder(tokens)
        hidden = self._encoder(embeddings, mask)

        # Embed entity shortlist
        shortlist_mask = get_text_field_mask(entity_shortlist)
        shortlist_embeddings = self._entity_embedder(entity_shortlist)

        # Embed entities
        entity_embeddings = self._entity_embedder(entity_ids)

        # Initialize losses
        entity_type_loss = 0.0
        entity_id_loss = 0.0
        vocab_loss = 0.0

        for timestep in range(sequence_length - 1):

            current_hidden = hidden[:, timestep]
            current_mask = mask[:, timestep]

            next_entity_types = entity_types[:, timestep + 1]
            next_entity_ids = entity_ids['entity_ids'][:, timestep + 1]
            next_shortlist_ids = entity_shortlist_ids[:, timestep + 1]
            next_tokens = tokens['tokens'][:, timestep + 1]

            # Predict entity types
            entity_type_logits = self._entity_type_projection(current_hidden[current_mask])
            entity_type_loss += F.cross_entropy(entity_type_logits,
                                                next_entity_types.long(),
                                                reduction='sum')

            # Predict entity id (from shortlist for now)
            projected_hidden = self._entity_id_projection(current_hidden)
            projected_hidden = projected_hidden.unsqueeze(2)
            entity_id_logits = torch.bmm(shortlist_embeddings, projected_hidden).squeeze(2)
            _entity_id_loss = F.cross_entropy(entity_id_logits,
                                              next_shortlist_ids,
                                              reduction='none')
            _entity_id_loss = _entity_id_loss * next_entity_types.float()  # Only predict for new entities
            entity_id_loss += _entity_id_loss.sum()

            # Now for the fun part - the next word.
            # TODO: Finish; right now we'll just do alias scores.
            copy_hidden = self._copy_mode_projection(current_hidden)  # TODO: Better name - hidden proj. for copy
            for i in range(batch_size):
                # Get the global and local alias arrays
                if next_entity_types[i] == 0:
                    continue
                import pdb; pdb.set_trace()
                next_entity_id = next_entity_ids[i]
                global_id, local_id = alias_database.lookup(next_entity_id.item())
                # Encode global ids
                global_id = global_id
                alias_id_mask = global_id.eq(0)
                global_token_embeddings = self._token_embedder({'tokens': global_id})
                alias_encoding, _ = self._alias_encoder(global_token_embeddings)
                alias_encoding = torch.tanh(alias_encoding)

                copy_scores = torch.einsum('ijk,k->ij', alias_encoding, copy_hidden[i, :])



        return {}

    def reset_states(self, batch_size: int) -> None:
        """Resets the model's internals. Should be called at the start of a new batch."""
        self._encoder.reset_states()
        self._state = None
