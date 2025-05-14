import argparse
import csv
import json
import logging
import requests
import random
import torch

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List
from context import *

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def clean_output(output):

    output = output.split('\n')[0].split('.')[0].strip()  # Remove content after newlines or periods

    # List of phrases to remove
    phrases_to_remove = ["Answer:", "answer:", "Response:", "response:","**Answer:**","**Answer:** ","*"]
    
    # Remove each phrase if it's present in the output
    for phrase in phrases_to_remove:
        output = output.replace(phrase, "").strip()
    
    # Additional cleaning for empty or irrelevant responses
    
    return output


def replace_single_with_double_quotes(text: str) -> str:
    # Replace occurrences of ' with "
    return text.replace("'", '"')

# Read jsonl file containing LM-KBC data
def read_lm_kbc_jsonl(file_path: str):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# Disambiguation baseline
def disambiguation_baseline(item):
    try:
        # If item can be converted to an integer, return it directly
        return int(item)
    except ValueError:
        # If not, proceed with the Wikidata search
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
            data = requests.get(url).json()
            # Return the first id (Could upgrade this in the future)
            return data['search'][0]['id']
        except:
            return item


# Read prompt templates from a CSV file
def read_prompt_templates_from_csv(file_path: str):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        prompt_templates = {row['Relation']: row['PromptTemplate'] for row in reader}
    return prompt_templates

# Read train data from a CSV file
def read_train_data_from_csv(file_path: str):
    with open(file_path, "r") as file:
        train_data = [json.loads(line) for line in file]
    return train_data

# Create a prompt using the provided data
def create_prompt(subject_entity_id: str,subject_entity: str, relation: str, prompt_templates: dict, instantiated_templates: List[str], tokenizer, few_shot: int = 0, task: str = "fill-mask") -> str:
    prompt_template = prompt_templates[relation]
    


    task_explanation = ("Please answer the question with your knowledge. Beforehand there a few examples. The output format shoudl be a "
                        "list of possible answers prefaced by 'Answer: ', also if there is no answer write Answer: ['']")
    
    if task == "text-generation":
        if few_shot > 0:
            random_examples = random.sample(instantiated_templates, min(few_shot, len(instantiated_templates)))
            few_shot_examples = "\n".join(random_examples)
            prompt = f"{task_explanation}\nDemonstrations:\n{few_shot_examples}\nQuestion: {prompt_template.format(subject_entity=subject_entity)}" 
       
       
        else:
            title = get_wikipedia_title(subject_entity_id)
            
            context = get_context_from_wikipedia(title)

             #just 0-shot prompt
            if relation == "BandHasMember":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the members, separated by ', ' with no extra text."
            elif relation == "CityLocatedAtRiver":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the river(s), separated by ', ' with no extra text."
            elif relation == "CompanyHasParentOrganisation":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the parent organization only or respond with '' if none, with no extra text."
            elif relation == "CountryBordersCountry":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the countrie(s), separated by ', ' with no extra text."
            elif relation == "CountryHasOfficialLanguage":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the language(s), separated by ', ' with no extra text."
            elif relation == "CountryHasStates":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the states / provinces, separated by ', ' with no extra text."
            elif relation == "FootballerPlaysPosition":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nProvide the position(s), separated by ', ' with no extra text."
            elif relation == "PersonCauseOfDeath":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nProvide only the cause, or respond with '' if unknown, with no extra text"
            elif relation == "PersonHasAutobiography":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the title, with no extra text."
            elif relation == "PersonHasEmployer":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the employer(s), separated by ', ' with no extra text."
            elif relation == "PersonHasNoblePrize":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the field only, or '' if none, with no extra text."
            elif relation == "PersonHasNumberOfChildren":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the number only."
            elif relation == "PersonHasPlaceOfDeath":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nProvide only the place, or respond with '' if unknown, with no extra text"
            elif relation == "PersonHasProfession":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the profession(s), separated by ', ' with no extra text."
            elif relation == "PersonHasSpouse":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the spouse name, with no extra text."
            elif relation == "PersonPlaysInstrument":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList the instrument(s), separated by ', ' with no extra text."
            elif relation == "PersonSpeaksLanguage":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList the language(s) separated by ', ' with no extra text."
            elif relation == "RiverBasinsCountry":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the country name, or '' if none, with no extra text."
            elif relation == "SeriesHasNumberOfEpisodes":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nAnswer with the number only."
            elif relation == "StateBordersState":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList only the state(s), separated by ', ' with no extra text."
            elif relation == "CompoundHasParts":
                prompt = f"Context:{context}\nQuestion:{prompt_template.format(subject_entity=subject_entity)}\nList the components, separated by ', ' with no extra text."
            
        

    else:
        prompt = prompt_template.format(subject_entity=subject_entity, mask_token=tokenizer.mask_token)
    return prompt

def run(args):
    # Load the model
    model_type = args.model
    print(model_type)
    logger.info(f"Loading the model \"{model_type}\"...")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    tokenizer.padding_side = 'left'  # Set padding to left for decoder-only models
    model = AutoModelForMaskedLM.from_pretrained(model_type)  if "bert" in model_type.lower()  else AutoModelForCausalLM.from_pretrained(model_type)
    task = "fill-mask" if "bert" in model_type.lower() else "text-generation"    
     
     # Set the pad token if it doesn't exist
    #if tokenizer.pad_token_id is None:
    #    tokenizer.pad_token_id = model.config.eos_token_id

    
    # Read the prompt templates and train data from CSV files
    if task == "text-generation":
        #pipe = pipeline(task=task, model=model, tokenizer=tokenizer, top_k=args.top_k, device=args.gpu, fp16=args.fp16) 
        # Initialize pipeline with required configurations
        pipe = pipeline(
            task=task,
            model=model,
            tokenizer=tokenizer,
            top_k=args.top_k,
            device=args.gpu,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
            padding=True  # Enables padding for batching
        )
        # Set the pad_token_id to EOS token if not set
        if model_type == "Qwen/Qwen2.5-1.5B-Instruct" or model_type == "Qwen/Qwen2.5-0.5B-Instruct" or model_type == "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B":
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
        else:
            pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]
        logger.info(f"Reading question prompt templates from \"{args.question_prompts}\"...")
        prompt_templates = read_prompt_templates_from_csv(args.question_prompts)
    else:
        pipe = pipeline(task=task, model=model, tokenizer=tokenizer, top_k=args.top_k, device=args.gpu) 
        logger.info(f"Reading fill-mask prompt templates from \"{args.fill_mask_prompts}\"...")
        prompt_templates = read_prompt_templates_from_csv(args.fill_mask_prompts)
    # Instantiate templates with train data
    instantiated_templates = []
    if task == "text-generation":
        logger.info(f"Reading train data from \"{args.train_data}\"...")
        train_data = read_train_data_from_csv(args.train_data)
        logger.info("Instantiating templates with train data...")
        for row in train_data:
            relation = row['Relation']
            prompt_template = prompt_templates[relation]
            object_entities = row['ObjectEntities']
            answers = ', '.join(object_entities)
            instantiated_example = prompt_template.format(subject_entity=row["SubjectEntity"]) + f" {answers}"
            instantiated_templates.append(instantiated_example)

    #print(instantiated_templates)

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = [json.loads(line) for line in open(args.input, "r")]
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = [create_prompt(
    subject_entity_id=row["SubjectEntityID"],
    subject_entity=row["SubjectEntity"],
    relation=row["Relation"],
    prompt_templates=prompt_templates,
    instantiated_templates=instantiated_templates,
    tokenizer=tokenizer,
    few_shot=args.few_shot,
    task=task,
    ) for row in input_rows]

    #print(prompts)

    # Run the model
    torch.cuda.empty_cache()
    logger.info(f"Running the model...")
    if task == 'fill-mask':
        outputs = pipe(prompts, batch_size=args.batch_size)
    else:
        outputs = pipe(prompts, batch_size=args.batch_size, max_new_tokens=20)

    torch.cuda.empty_cache()

    print("Starting the results")
    results = []
    for row, output, prompt in zip(input_rows, outputs, prompts):
        torch.cuda.empty_cache()
        object_entities_with_wikidata_id = []
        if task == "fill-mask":
            for seq in output:
                if seq["score"] > args.threshold:
                    wikidata_id = disambiguation_baseline(seq["token_str"])
                    object_entities_with_wikidata_id.append(wikidata_id)
        else:
            # Remove the original prompt from the generated text
            qa_answer = output[0]['generated_text'].split(prompt)[-1].strip()
            qa_answer = clean_output(qa_answer)
            qa_answer = replace_single_with_double_quotes(qa_answer)
            
            
            if row["Relation"] == "PersonHasNumberOfChildren" or row["Relation"] == "SeriesHasNumberOfEpisodes":
                
                result_row = {
                    "SubjectEntityID": row["SubjectEntityID"],
                    "SubjectEntity": row["SubjectEntity"],
                    "ObjectEntitiesID": str(qa_answer),
                    "Relation": row["Relation"],
                }
            else:
                qa_entities = qa_answer.split(", ")
                for entity in qa_entities:
                    wikidata_id = disambiguation_baseline(entity)
                    object_entities_with_wikidata_id.append(wikidata_id)
                
                result_row = {
                    "SubjectEntityID": row["SubjectEntityID"],
                    "SubjectEntity": row["SubjectEntity"],
                    "ObjectEntitiesID": object_entities_with_wikidata_id,
                    "Relation": row["Relation"],
                }
                

        
        results.append(result_row)

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Model with Question and Fill-Mask Prompts")
    parser.add_argument("-m", "--model", type=str, default="bert-base-cased", help="HuggingFace model name (default: bert-base-cased)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input test file (required)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file (required)")
    parser.add_argument("-k", "--top_k", type=int, default=10, help="Top k prompt outputs (default: 100)")
    parser.add_argument("-t", "--threshold", type=float, default=0.1, help="Probability threshold (default: 0.1)")
    parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU ID, (default: -1, i.e., using CPU)")
    parser.add_argument("-qp", "--question_prompts", type=str, required=True, help="CSV file containing question prompt templates (required)")
    parser.add_argument("-fp", "--fill_mask_prompts", type=str, required=True, help="CSV file containing fill-mask prompt templates (required)")
    parser.add_argument("-f", "--few_shot", type=int, default=5, help="Number of few-shot examples (default: 5)")
    parser.add_argument("--train_data", type=str, required=True, help="CSV file containing train data for few-shot examples (required)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the model. (default:32)")
    parser.add_argument("--fp16", action="store_true", help="Enable 16-bit model (default: False). This is ignored for BERT.")

    args = parser.parse_args()

    run(args)
