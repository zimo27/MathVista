import argparse
import logging
import os
import re

from openai import AzureOpenAI
from rich.logging import RichHandler
from tqdm import tqdm

from evaluation.prompts.ext_ans import demo_prompt
# from models import gpt
from utilities import read_json, save_json


def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(model, response, problem, quick_extract=False):
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # quick extraction
    if quick_extract:
        logging.info("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = model.get_response(user_prompt=full_prompt)
        return extraction
    except Exception as e:
        logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logging.info(e)

    return ""


def parse_args():
    parser = argparse.ArgumentParser()
    # input
    parser.add_argument('--results_file_path', type=str, default='answer.json')
    parser.add_argument('--response_label', type=str, default='response', help='response label for the input file')
    parser.add_argument('--max_num_problems', type=int, default=-1, help='The max number of problems to run')
    parser.add_argument('--quick_extract', action='store_true', help='use rules to extract answer for some problems')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer extraction')
    # output
    parser.add_argument('--save_every', type=int, default=100, help='save every n problems')

    parser.add_argument('--openai_api_key', type=str, default=os.getenv("OPENAI_API_KEY"), help='OpenAI API key')
    parser.add_argument('--openai_model', type=str, default="gpt-4o", help='OpenAI model name (e.g., gpt-3.5-turbo, gpt-4)')


    args = parser.parse_args()
    return args


def main():
    logging.info("MathVista: Extract Answers - Start")
    args = parse_args()

    # args
    label = args.response_label

    # import openai

    # # Set OpenAI API key
    # openai.api_key = os.getenv("OPENAI_API_KEY")

    # assert openai.api_key is not None, "Env var OPENAI_API_KEY is not set but is required for OpenAI client."

    # class GPT_Model:
    #     def __init__(self, model="gpt-4o"):
    #         self.model = model

    #     def get_response(self, user_prompt):
    #         try:
    #             response = openai.ChatCompletion.create(
    #                 model=self.model,
    #                 messages=[{"role": "user", "content": user_prompt}],
    #                 temperature=0.7,
    #             )
    #             return response["choices"][0]["message"]["content"]
    #         except Exception as e:
    #             logging.error(f"Error while generating response: {e}")
    #             return ""
    import openai

    # Set OpenAI API Key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    class GPT_Model:
        def __init__(self, model="gpt-4o"):
            self.model = model

        def get_response(self, user_prompt):
            try:
                response = openai.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt},
                    ],
                    model=self.model,
                    max_tokens=512,
                    temperature=0.7,
                )
                return response.choices[0].message.content
            except Exception as e:
                logging.error(f"Error while generating response: {e}")
                return ""


    # Initialize the model
    model = GPT_Model(model="gpt-4o")


    logging.info(f"Reading {args.results_file_path}...")
    results = read_json(args.results_file_path)

    full_pids = list(results.keys())

    skip_pids = []
    for pid, problem in results.items():
        extraction = problem.get('extraction')
        if extraction is not None and verify_extraction(extraction):
            skip_pids.append(problem['pid'])

    if args.rerun:
        test_pids = full_pids
    else:
        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
            )
        test_pids = [pid for pid in full_pids if pid not in skip_pids]

    if args.max_num_problems > 0:
        test_pids = test_pids[: min(args.max_num_problems, len(test_pids))]
        logging.info(f'Limiting number of problems to {args.max_num_problems}.')

    logging.info(f"Number of test problems to run: {len(test_pids)}")

    for i, pid in enumerate(tqdm(test_pids)):
        problem = results[pid]

        assert label in problem
        response = problem[label]
        extraction = extract_answer(model, response, problem, args.quick_extract)
        results[pid]['extraction'] = extraction

        if (i % args.save_every == 0 and i > 0) or i == len(test_pids) - 1:
            save_json(results, args.results_file_path)
            logging.info(f"Saved results to {args.results_file_path}")

    logging.info("MathVista: Extract Answers - Finish")


if __name__ == '__main__':
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                markup=False,
                show_path=False,
                omit_repeated_times=False,
            )
        ],
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()
