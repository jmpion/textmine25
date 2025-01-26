import torch
import torch.nn as nn
from transformers import AutoModel

def extract_and_pool_embeddings(token_ids, token_embeddings):
    # Create a mask where token_ids are non-zero (valid, not padding)
    mask = (token_ids != 0)
    assert torch.any(token_ids != 0), print("No valid token IDs found.")
    
    # Gather embeddings using valid token IDs
    span_embeds = token_embeddings.gather(1, token_ids.unsqueeze(-1).expand(-1, -1, token_embeddings.size(-1)))
    
    # Set embeddings of padding tokens to -inf so they are ignored in max-pooling
    span_embeds = span_embeds.masked_fill(~mask.unsqueeze(-1), float('-inf'))
    
    # Max-pooling across tokens, excluding padded tokens (which are now -inf)
    maxpool = torch.max(span_embeds, dim=1).values
    # maxpool = torch.logsumexp(span_embeds, dim=1)

    # Apply the dense layer to the pooled embeddings
    return maxpool

class SpERT(nn.Module):
    def __init__(self, bert_model_name, hidden_size=768, num_labels=2, cache_dir=None):
        super(SpERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name, hidden_dropout_prob=0.3, cache_dir=cache_dir)
        self.hidden_size = hidden_size # typically 768

        # Classification layers
        self.classifier = nn.Sequential(
                                nn.Linear(1 * hidden_size, num_labels), # we add a head for entity classification.
        )

        red_dim = 10
        self.proj_s = nn.Linear(hidden_size, red_dim)
        self.proj_o = nn.Linear(hidden_size, red_dim)
        self.proj_c = nn.Linear(hidden_size, red_dim)

        prelogits_dim = 100
        self.bilinear_sc = nn.Bilinear(red_dim, red_dim, prelogits_dim, bias=True)
        self.bilinear_oc = nn.Bilinear(red_dim, red_dim, prelogits_dim, bias=True)
        self.bilinear_so = nn.Bilinear(red_dim, red_dim, prelogits_dim, bias=True)
        self.fc = nn.Linear(3 * prelogits_dim, num_labels)

    def resize_token_embeddings(self, num_tokens):
        self.bert.resize_token_embeddings(num_tokens)

    def forward(
            self, 
            input_ids, 
            attention_mask, 
            subject_token_ids, 
            object_token_ids, 
            subject_token_ids_end,
            object_token_ids_end,
            relation_token_ids=None,
            labels=None,
            ):
        # Get the BERT embeddings of the input sequence.
        # The output is a matrix-like object, where each row corresponds to a token in the inputs sequence,
        # and each column corresponds to a hidden component.
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = bert_outputs.last_hidden_state  # Shape: (batch_size, seq_length, hidden_size)

        # Extract token embeddings and apply max-pooling for subject and object spans
        subject_span_maxpool = extract_and_pool_embeddings(subject_token_ids, token_embeddings)
        object_span_maxpool = extract_and_pool_embeddings(object_token_ids, token_embeddings)
        relation_span_maxpool = extract_and_pool_embeddings(relation_token_ids, token_embeddings)

        # Sizes: (batch_size, hidden_dim) for all maxpools

        # Dimension reduction, by projection.
        subject_span_maxpool = self.proj_s(subject_span_maxpool)
        object_span_maxpool = self.proj_o(object_span_maxpool)
        relation_span_maxpool = self.proj_c(relation_span_maxpool)

        # Get interactions subject/object, subject/context, object/context.
        interact_so = nn.Tanh()(self.bilinear_so(subject_span_maxpool, object_span_maxpool))
        interact_sc = nn.Tanh()(self.bilinear_sc(subject_span_maxpool, relation_span_maxpool))
        interact_oc = nn.Tanh()(self.bilinear_oc(object_span_maxpool, relation_span_maxpool))

        # Concatenate interactions representations.
        interacts = torch.cat((interact_so, interact_sc, interact_oc), dim=-1)

        # Classify.
        logits = self.fc(interacts)

        # Calculate loss (optional: if labels are provided)
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()  # or another loss function
            loss = loss_fn(logits, labels)
            return loss, logits
        
        return logits
