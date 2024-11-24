import torch
from torch import nn
from transformers import RobertaTokenizer, BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


class RumorDetectionModel(nn.Module):
    def __init__(self, encoder, rgcn_model, explanation_model, fc_model, args,
                 decoder_model_name):
        super(RumorDetectionModel, self).__init__()
        self.args = args
        self.encoder = encoder
        self.rgcn_model = rgcn_model
        self.explanation_model = explanation_model
        self.fc_model = fc_model
        self.tokenizer = RobertaTokenizer.from_pretrained(args.bert)
        self.decoder_tokenizer = BartTokenizer.from_pretrained(decoder_model_name)
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_model_name)
        self.projection = nn.Linear(explanation_model.fc.out_features, self.decoder.config.d_model)

    def forward(self, data):
        device = next(self.parameters()).device

        tweet_texts = data.tweet_text
        evidence_texts = data.evidence_text

        all_texts = []
        tweet_indices = []
        evidence_indices = []

        idx = 0
        for i, (tweet, evidences) in enumerate(zip(tweet_texts, evidence_texts)):
            all_texts.append(tweet)
            tweet_indices.append(idx)
            idx += 1

            for evidence in evidences:
                all_texts.append(evidence)
                evidence_indices.append(idx)
                idx += 1

        inputs = self.tokenizer(
            all_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        embeddings = self.encoder(inputs['input_ids'], inputs['attention_mask'])

        if hasattr(embeddings, 'last_hidden_state'):
            x = embeddings.last_hidden_state
        else:
            x = embeddings

        edge_index = data.edge_index.to(device)
        edge_type = data.edge_type.to(device)
        batch = data.batch.to(device)

        x = self.rgcn_model(x, edge_index, edge_type)

        batch_size = len(tweet_texts)
        tweet_features = x[tweet_indices]

        evidence_features_list = []
        idx = 0
        for evidences in evidence_texts:
            num_evidences = len(evidences)
            if num_evidences > 0:
                evidence_indices_batch = evidence_indices[idx:idx + num_evidences]
                evidence_features = x[evidence_indices_batch]
                idx += num_evidences
            else:
                evidence_features = torch.zeros((1, x.size(-1)), device=device)
            evidence_features_list.append(evidence_features.unsqueeze(0))

        max_evidences = max([evi.size(1) for evi in evidence_features_list]) if evidence_features_list else 1
        padded_evidence_features = []
        for evi in evidence_features_list:
            num_evidences = evi.size(1)
            if num_evidences < max_evidences:
                padding = torch.zeros((1, max_evidences - num_evidences, x.size(-1)), device=device)
                evi = torch.cat([evi, padding], dim=1)
            padded_evidence_features.append(evi)
        evidence_features = torch.cat(padded_evidence_features, dim=0)

        explanations = self.explanation_model(evidence_features)

        combined_features = torch.cat([tweet_features, explanations], dim=1)
        logits = self.fc_model(combined_features)

        decoder_input = self.projection(explanations)

        encoder_hidden_states = decoder_input.unsqueeze(1)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)

        encoder_attention_mask = torch.ones((encoder_hidden_states.size(0), encoder_hidden_states.size(1)),
                                            device=device)

        generated_ids = self.decoder.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_attention_mask,
            max_length=50,
            num_beams=5,
            early_stopping=True
        )
        generated_texts = self.decoder_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return logits, generated_texts