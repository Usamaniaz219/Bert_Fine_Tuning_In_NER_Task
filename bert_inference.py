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
    # test_sentence = (
    #  """5:100 sf
    # "5:100 sf"
# 5/100 sf
# "<= 100 beds: 1 per bed  101—300 beds: 1.1 per bed  301—500 beds: 1.2 per bed  > 500 beds: 1.3 per bed"
# "1 for every 4 seats or 100 square feet of gross floor area, whichever is greater, at maximum capacity"
# "Same as required for office buildings"
# "0.01 gross square feet"
# "For purposes of administering the foregoing hotel parking ratios, 'guest rooms' shall refer to individual lodging units, regardless of the actual number of rooms in such unit. Thus, a lodging unit with 2 separate sleeping rooms, or separate sleeping and living rooms, shall be considered to be 1 guest room."
# "1 per lodging room + 1 per 5 seats  3"
# "4 spaces for facilities with 5 or less residents, the parking spaces may be arranged 1 behind the other; and 4 spaces plus 1 space per each 5 beds"
# "2 spaces per dwelling unit plus 1 additional space for each bedroom exceeding 2 bedrooms. For buildings with 2 dwelling units or less, the 3 and 4 spaces, when required, can be in tandem with the 1 2 spaces required"
# "1 space for each 6.5 feet of linear pew or 3.5 seats in the auditorium; provided, however, that where a church building is designed or intended to be used by 2 congregations or special meetings, an additional 50% of the minimum spaces required shall be provided."
# "5 spaces minimum, or 5 percent of the total site area excluding the landscaped areas, whichever is greater."
# "2 spaces for the dwelling unit, plus 2 spaces for visitors, the parking spaces may be arranged 1 behind the other."
# "2 spaces for each 3 bedroom unit,15 for each 2 bedroom unit,1 space for each 1 bedroom unit"
# "Primary unit  must have 2 garage spaces;  2 unit:  1 garage space for 1 bedroom or studio; 2 garage spaces for  2 or more bedrooms."
# "7.0 spaces per 1,000 sf for standard or carry out restaurants, 3 spaces per 1,000  for fast food."
# "Parking space for all vehicles used in conjunction with the business; and 1 space  for each 2 employees on the maximum (most workers) working shift or 1 space for each 350 sq.  ft. for the 1 10,000 sq. ft. of gross floor area, and 1 space for each 500 sq. ft. for the next 40,000 sq. ft. of gross floor area, and 1 space for each 1,000  sq. ft. for the next 50,000 sq. ft. of gross floor area and 1 space for each 2,000  sq. ft. for all floor area over 100,000 sq. ft. of gross floor area, whichever is  the greater"
# "Space shall be provided on-site for 5 vehicles to circulate and to deposit recyclable  materials and 1 parking space shall be provided for each employee on the largest shift  and 1 space for each commercial vehicle associated with the use"
# "1 space for each  employee, plus, 1 space for each 5 children, or 1 space for  each 10 children where a circular driveway or its  equivalent, designed for the continuous flow of passenger  vehicles for the purpose of loading and unloading children  and capable of simultaneously accommodating at least 2 such  vehicles, is provided on the site."
# "Space shall be provided on-site for 10 vehicles to circulate and to deposit recyclable  materials and 1 parking space shall be provided for each employee on the largest shift  and 1 space for each commercial vehicle associated with the use"
# "1 for each 10,000 square feet of lot area for the 1 1 acre of property for single-operator  and multiple-operator lots. A minimum of 5 parking spaces shall be provided for each  operator on each parcel regardless of the lot size."
# "15 spaces for every  studio or 1-bedroom unit; 2 spaces for every unit with 2 or  more bedrooms and 1 additional guest space for 4 units and  every 4 thereafter. At least 50% of the spaces must be  covered. Of the covered and uncovered spaces, 50% of each  may be compact-sized. Exception: (1) Artist’s joint living  and working 25s need not provide covered spaces; (2)  The city may reduce or waive parking requirements for  housing projects with units committed to long-term,  low-income or senior citizen’s housing, e.g., as defined  under the Federal Government Section 8 Housing or its  equivalent."
# "1 space for each  employee, but not less than 2 spaces for such facility."
# "1 space for each 6 seats or 12 feet of bench in principal place of worship."
# "3.0 spaces without car wash; 4.0 spaces with car wash"
# "1 space for each ADU in addition to the required spaces for Single Family Detached  Dwellings (1. a.)"
# "1 or 2 persons (other than family members): no spaces; 3 to 5 persons (other than  family members): 1.0 space; 6 to 8 persons (other than family members) 2.0 spaces."  # problem in extracting correct entities
# "1 space per 3 seats plus 1 per each 2 employees plus reservoir lane capacity equal  to 5 spaces per drive-in window plus 5 spaces designated for the ordering station."
# "1 space per 300 square feet of area within main building plus reservoir land capacity  equal to 3 spaces per window (10 spaces if window serves 2 stations)."
# "1 for every 3 beds or 2 rooms, whichever is less, and 1 for each employee on duty based upon maximum employment shift."
# "15 per each efficiencyor 1-bedroom dwelling unit, and 2 per each unit with 2 or more bedrooms."
    # )


    test_sentence = (
        # "A developer is submitting plans for a boutique hotel featuring twelve (12) guest suites, a 180 square meter conference facility, and a rooftop lounge."  # Failed to extract any entity
        # "The municipal code stipulates that hotels must provide one and one-tenth (1.1) spaces for each sleeping room, plus one (1) space for every seven (7) square meters of assembly area." 
        # "What is the minimum number of off-street parking spaces required?"
        # "The facility will feature a five-thousand (5,000) square foot public reading area, a community meeting room with one hundred (100) fixed seats, and an eight-hundred (800) square foot administrative wing. " # problem is in numwords_2_num package or module for not converting the words into number
        # "The zoning ordinance requires one (1) space per 400 square feet of public area, one (1) space for every four (4) fixed seats, and one (1) space per 300 square feet of staff area. Determine the total parking requirement."

        # "A craft distillery with a tasting room is proposed. The operation includes a four-hundred (400) square meter production area and a 150 square meter visitor center. "
        # "The local bylaws mandate one (1) parking space per three hundred (300) square meters of industrial floor area and one (1) space per fifteen (15) square meters of customer-facing area. Calculate the total number of spaces needed."
        # "A mixed-use development is planned with eight thousand (8,000) square feet of ground-floor retail and twelve (12) residential apartments above. "  # problem is in numwords_2_num
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







