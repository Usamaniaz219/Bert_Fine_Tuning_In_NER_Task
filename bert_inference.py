import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import json
from numwords_to_nums.numwords_to_nums import NumWordsToNum

# Load tokenizer and model
model_path = "results/checkpoint-166"
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

# Load label mapping
with open(f"{model_path}/id_to_label.json", "r") as f:
    id_to_label = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(sentence):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True
    )
    offset_mapping = inputs.pop("offset_mapping")[0]  # shape: (seq_len, 2)
    special_tokens_mask = inputs.pop("special_tokens_mask")[0]
    
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
    input_ids = inputs["input_ids"][0].cpu().numpy()
    
    results = []

    for idx, (pred_id, input_id, offset, is_special) in enumerate(zip(predictions, input_ids, offset_mapping, special_tokens_mask)):
        if is_special or offset[0] == offset[1]:
            continue  # Skip [CLS], [SEP], and padding tokens

        token = tokenizer.convert_ids_to_tokens(int(input_id))
        label = id_to_label[str(pred_id)]
        start, end = offset.tolist()

        results.append({
            "token": token,
            "label": label,
            "start": start,
            "end": end
        })
    
    return results

def extract_entities_with_spans(predicted_tokens):
    entities = []
    current_tokens = []
    current_label = None
    start_index = None

    for token_info in predicted_tokens:
        label = token_info["label"]
        token = token_info["token"]
        start = token_info["start"]
        end = token_info["end"]

        if label != 'O':
            if current_label == label:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "text": tokenizer.convert_tokens_to_string(current_tokens).strip(),
                        "label": current_label,
                        "start": start_index,
                        "end": prev_end
                    })
                current_tokens = [token]
                current_label = label
                start_index = start
        else:
            if current_tokens:
                entities.append({
                    "text": tokenizer.convert_tokens_to_string(current_tokens).strip(),
                    "label": current_label,
                    "start": start_index,
                    "end": prev_end
                })
                current_tokens = []
                current_label = None
                start_index = None
        prev_end = end  # keep track of previous token's end

    if current_tokens:
        entities.append({
            "text": tokenizer.convert_tokens_to_string(current_tokens).strip(),
            "label": current_label,
            "start": start_index,
            "end": prev_end
        })

    return entities



def numwordTonums(text):
    num = NumWordsToNum()
    result = num.numerical_words_to_numbers(text,convert_operator=False,calculate_mode=False,evaluate=False)
    return result

if __name__ == "__main__":
    test_sentence = (
        "The code requires retail to provide one (1) space per 350 square feet. For the residential units, the requirement is one and eight-tenths (1.8) spaces per dwelling unit.What is the compound parking requirement for the entire building?"
    )

    # test_sentences = test_sentences.split("\n")
    # for i, test_sentence in enumerate(test_sentences):
    test_sentence = numwordTonums(test_sentence)
    # print(test_sentence)
    token_predictions = predict(test_sentence)
    entities = extract_entities_with_spans(token_predictions)
    print("test_sentence : ", test_sentence)
    print("Extracted Entities with Positions:")
    if entities:
        for ent in entities:
            print(ent)
    else:
        print("[]")







