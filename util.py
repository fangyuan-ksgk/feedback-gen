import requests
import re
import aiohttp
from openai import OpenAI
import os
import json

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
)

def get_response(prompt):
    completion = client.chat.completions.create(
        model="gpt-4-turbo-2024-04-09",
        messages=[
            {'role': 'system', 'content': "You are a helpful assistant that always closely follows instructions."},
            {'role': 'user', 'content': prompt}
        ],
    )
    response = completion.choices[0].message.content
    return response

# Path to transcript, modify here
def read_transcript(transcript_dir):
    with open(transcript_dir, 'r', encoding='utf-8') as file:
        transcript = file.read()
    return transcript

# Parse the transcript string to extract meaningful information
def parse_transcripts(transcripts):
    parsed_transcripts = []
    for line in transcripts.split('\n'):
        if line.startswith('Customer:') or line.startswith('Agent:'):
            parsed_transcripts.append(line.strip())
    return parsed_transcripts


def clean_up_transcript(transcript_log):
    PARSE_TRANSCRRIPT_PROMPT = PARSE_TRANSCRRIPT_TEMPLATE.format(scattered_transcript=transcript_log)
    transcripts = get_response(PARSE_TRANSCRRIPT_PROMPT)
    parsed_transcripts = parse_transcripts(transcripts)
    return parsed_transcripts

# Load & Clean up on the transcript
def load_transcript(transcript_dir):
    clean_transcripts = []
    max_iter = 5
    iter = 0
    while len(clean_transcripts) == 0 and iter < max_iter:
        transcripts = read_transcript(transcript_dir)
        clean_transcripts = clean_up_transcript(transcripts)
        iter += 1
    return clean_transcripts


# Use the LLM to do the parsing, current version is dumb | and the transcript is not that pretty anyways
PARSE_TRANSCRRIPT_TEMPLATE = """Here is a transcript of a conversation between a customer and an insurance agent. The transcript is scattered and one utterance from Agent or Customer could be split into multiple lines. Help me clean it up and provide a formatted transcript. Result should be formated as 'Customer: <sentence> \nAgent: <sentence>' where each sentence is a coherent utterance. 
[SCATTERED TRANSCRIPT] \n {scattered_transcript} [END]
[EXAMPLE]
Scattered Transcript
((2024-04-25 07.28.37.334718)) CUSTOMER: well, thanks for asking!
((2024-04-25 07.28.38.765250)) CUSTOMER: How about you, how's everything going on your end?
((2024-04-25 07.28.57.326295)) AGENT: I think it's excellent and you know, 
((2024-04-25 07.32.51.316295)) AGENT: I've been just so happy with the fact that I've been make been able to make a difference in so many lives in such a short time and I'm hoping I can help you in in any way possible in your future plans.
Cleaned Transcript
Customer: well, thanks for asking!  How about you, how's everything going on your end?
Agent: I think it's excellent and you know, I've been just so happy with the fact that I've been make been able to make a difference in so many lives in such a short time and I'm hoping I can help you in in any way possible in your future plans.
[END]

Scattered Transcript
{scattered_transcript}
Cleaned Transcript
"""


agent_competency = {
    "1. Product_and_Industry_Knowledge": {
        "1.1.1": "Did_AGENT_use_any_wrong_insurance_terms (reverse)",
        "1.1.2": "Did_AGENT_explain_the_value_of_insurance_in_mitigating_risk",
        "1.2.1": "Did AGENT explain coverage of the policy?",
        "1.2.2": "Did AGENT explain features of the policy?",
        "1.2.3": "Did AGENT give any wrong response regarding policy exclusions? (reverse)",
        "1.2.4": "Did AGENT compare product features with those from competitors when necessary?",
        "1.2.5": "Did AGENT explain how the product fits CUSTOMER's unique situation?",
        "1.3.1": "Did AGENT give an example or analogy of how the policy is in line with current social/economic/market situation?",
    },
    "2. Customer Relationship Management": {
        "2.1.1": "Did AGENT open the conversation with professional and friendly greeting?",
        "2.1.2": "Did AGENT initial small talk or show genuine interest in CUSTOMER to build rapport?",
        "2.2.1": "Did AGENT ask questions to elicit sharing about CUSTOMER background?",
        "2.2.2": "Did AGENT ask questions to elicit sharing about CUSTOMER needs?",
        "2.2.3": "Did AGENT do active listening by asking follow-up question about CUSTOMER's background?",
        "2.2.4": "Did AGENT do active listening by paraphrasing or summarizing what CUSTOMER shared?",
        "2.3.1": "Did AGENT demonstrate personalization by referencing customer-specific information previously shared?",
        "2.3.2": "Did AGENT mention that there will be future follow-up or check-in?",
        "2.4.1": "Did AGENT make explicit commitments to service quality and availability for future assistance?",
        "2.4.2": "Did AGENT give an example of how AGENT helped another customer solve a problem in the past?",
        "2.4.3": "Did AGENT explain how CUSTOMER can get help when needed?",
        "2.5.1": "Did AGENT actively ask for feedback?",
        "2.5.2": "Did AGENT respond constructively to feedback?",
    },
    "3. Negotiation and Sales Skills": {
        "3.1.1": "Did AGENT explain USP (Unique Selling Point) of policy compared to competitor?",
        "3.1.2": "Did AGENT link policy benefits with CUSTOMER's particular situation?",
        "3.2.1": "Did AGENT acknowledge all CUSTOMER objections and concerns with empathy?",
        "3.2.2": "Did AGENT propose a solution or alternative to all CUSTOMER objections and concerns?",
        "3.3.1": "Did AGENT check CUSTOMER readiness to buy before the conversation ended?",
        "3.3.2": "Did AGENT summarize key benefits linked to client needs before the conversation ended?",
        "3.4.1": "Did AGENT state the next steps after this conversation?",
        "3.4.2": "Did AGENT ask questions to assess potential to up-sell or cross-sell?",
    },
    "4. Communication Skills": {
        "4.1.1": "Did AGENT use examples or analogies to explain complex insurance terms?",
        "4.1.2": "Did AGENT ask at least one question to check CUSTOMER's understanding?",
        "4.2.1": "Throughout the whole transcript, did AGENT keep explanation succinct without unnecessary details?",
        "4.2.2": "Did AGENT emphasize key points without unnecessary repetition?",
        "4.2.3": "Did AGENT summarize lengthy discussions?",
        "4.3.1": "Did AGENT adjust communication style and language use to CUSTOMER's style and language?",
        "4.3.2": "Did AGENT quickly address all misunderstandings if any?",
        "4.3.3": "Was AGENT sensitive to cultural norms?",
        "4.4.1": "Did AGENT use words like 'I See' or 'I Understand' to show active listening?",
        "4.4.2": "Did AGENT reflect or paraphrase CUSTOMER statements?",
    },
    "5. Analytical Skills": {
        "5.1.1": "Did AGENT point out issues or gaps in CUSTOMER's current insurance coverage if CUSTOMER has existing insurance coverage?",
        "5.1.2": "Did AGENT identify and question assumptions that may affect CUSTOMER's understanding or actions?",
    },
}


PROMPT_TEMPLATE = """<s>[INST]{inst}[/INST]{example}</s><s>[INST]{inst}[/INST]{text}\n\n<OUTPUT>\n\n{query}\n"""

fastModelLevel1PromptGpt = """<task>
- You are a top insurance sales agents in Philippines. 
- You are tasked to review a conversation conducted between a human trainee insurance sales AGENT and a potential CUSTOMER in their first sales meeting and be an assessor. 
- Your overarching objective is to assess the AGENT's performance during the conversation, and provide actionable feedback for AGENT learning.  
- Do this by following the <steps>. Take a deep breath and do this task step by step. 
</task>

<steps>
- Steps to achieve the objective in the <task>: 
- Step 1: for each atom, give one sentence explanation of why it was a "yes" or "no". 
- Step 2: Then give one example of what the AGENT said in the transcript that can be said differently to change a "No" to a "Yes"
</steps>

{additional_information}
Give your feedback in the following format and NOTHING ELSE:

<OUTPUT>

[Numbering for atom] [atom detail]
ASSESSMENT(Change all ASSEASSMENT to ASSESSMENT): [Based on how AGENT did in transcript, RESPOND ONLY WITH "Yes" or "No". There is no "partial" cases.]
REASON: [Reason for giving a "Yes" or "No"]
EXTRACT: [Extract one exact line from transcript that gave a "Yes" or "No"]
IMPROVEMENT: [Provide one example that can be replaced in the conversation transcript that can change a "No" or a "Yes" or a "Yes" to a "No" if atom is reverse]

<END>
"""

EXAMPLE = """
AGENT: Hey there!
CUSTOMER: Hey! Nice to meet you, nice spot for a coffee huh?
AGENT: Yes I like the chill vibe here. So, are you holding any insurance policies in place?
CUSTOMER: Oh I thought we could chat for a bit longer. Yeah I guess I have a policy, but you know its been a while since I looked into it. Do you usually get straight to business with your clients?
AGENT: Oh yes, I always talk business then relax later

<OUTPUT>

1.1.2 Did AGENT explain the value of insurance in mitigating risk?
ASSESSMENT: No 
REASON: When customer talked about policy, you did not provide any information about insurance mitigating risk. You also didn't add it on later on during the conversation so customer does not know the value of your insurance policy comapared to his. 
EXTRACT: "Oh yes, I always talk business then relax later"
IMPROVEMENT: Instead of saying "Oh yes, I always talk business then relax later", maybe you can try saying "FWD's Set for Life is a plan for savings and safety till you're 100. You can adjust it to save for retirement or your kid's school. It lets you invest safely or with risks, and gives you a bonus every five years after the first 10. You can also add more protection for illness or worse cases. It's a way to ensure you're covered for the future!"

<END>
"""

product_info = """
Refer to <productinformation> to check the conversation transcript against policy details. 
<productinformation>
    Companies:
  FWD:
    Product Name: "Set for Life"
    Product Type: "Limited Pay VUL"
    Premium Payment Period: "5, 7, 10"
    Currency: "PHP"
    Minimum Premium: "5Pay - PHP 40k, 7Pay - PHP 24k, 10Pay - PHP 20k"
    Minimum SA: "500% of regular annual premium"
    SA Flexibility: "Yes, with SA Multiplier"
    Issue Age: "0 to 70 years old"
    Insurance Coverage: "Up to age 100"
    Death Benefit: "Higher of (SA, AV)"
    Underwriting: "Full Underwriting"
    Mandatory Riders: "Accident, WP"
    Optional Riders: "Term Rider, CI, HIB"
    Commission: "5 Pay - 30%/10%/5%, 7 Pay - 35%/10%/5%, 10 Pay - 40%/15%/5%/5%/5%"
    Premium Charge: "5Pay -70%/45%/0%, 7Pay - 75%/50%/10%/10%, 10Pay - 90%/60%/35%/0%"
    Policy Fee: "n/a"
    Premium Holiday Charge: "None"
    Premium Extension Bonus: "n/a"
    Surrender Charge: "None"

  AXA Life:
    Product Name: "My Life Choice"
    Product Type: "Limited Pay VUL"
    Premium Payment Period: "7, 10"
    Currency: "PHP and USD"
    Minimum Premium: "7 pay - PHP 30k, 10 pay - PHP 20k"
    Minimum SA: "7 pay - PHP 210k, 10 pay - PHP 200k"
    SA Flexibility: "Yes, with SA Multiplier"
    Issue Age: "0 to 70 years old"
    Insurance Coverage: "Up to age 100"
    Death Benefit: |
      DB1: AV + SA
      DB2: Higher of (SA, AV)
    Underwriting: "Full Underwriting"
    Mandatory Riders: "Payor's Clause, WP, ADDD"
    Optional Riders: "Term Rider, CI, HIB, Bright Rider"
    Commission: "7 pay - 25%/5%/5%, 10 pay - 35%/10%/5%/5%/5%"
    Premium Charge: "7 Pay - 40%/35%/35%, 10 Pay - 40%/40%/35%/35%/35%"
    Policy Fee: "n/a"
    Premium Holiday Charge: "Same as Premium Charge"
    Premium Extension Bonus: "n/a"
    Surrender Charge: "All Variants"

  Pru Life:
    Product Name: "Prulink Exact Protector"
    Product Type: "Limited Pay VUL"
    Premium Payment Period: "5, 7, 10, 15"
    Currency: "PHP and USD"
    Minimum Premium: "PEP -PHP45k / USD600, PEP-LP - PHP 22k"
    Minimum SA: "500% of regular annual premium"
    SA Flexibility: "Yes, with SA Multiplier"
    Issue Age: "0 to 70 years old"
    Insurance Coverage: "Up to age 100"
    Death Benefit: "AV + SA"
    Underwriting: "Full Underwriting"
    Mandatory Riders: "ATPD, ADD"
    Optional Riders: "Future Safe, PA, Payor Term, Payor Waiver, WPTPD, CCB, LCB, LC Plus, LC Advance Plus, Multiple Life Care Plus, LCW, Hospital Income"
    Commission: "5 pay - 35%/15%/5%, 7pay - 35%/15%/15%, 10 pay -  37.50%/15%/5%/5%/5%, 15 pay - 40%/15%/5%/5%/5%"
    Premium Charge: "5 pay - 90%/55%, 7pay - 90%/55%, 10 pay - 93%/70%, 15 pay - 100%/65%/25%"
    Policy Fee: "n/a"
    Premium Holiday Charge: "None"
    Premium Extension Bonus: "n/a"
    Surrender Charge: "None"

  Sunlife Prime:
    Product Name: "Sun MaxiLink Prime"
    Product Type: "Limited Pay VUL"
    Premium Payment Period: "10"
    Currency: "PHP"
    Minimum Premium: "PHP 15,225"
    Minimum SA: "PHP350k"
    SA Flexibility: "No"
    Issue Age: "0 to 65 years old"
    Insurance Coverage: "Up to age 88"
    Death Benefit: "AV + SA"
    Underwriting: "Full Underwriting"
    Mandatory Riders: "None"
    Optional Riders: "ADB, TDB, WPD, WPDD, HIB, CIB"
    Commission: ""
    Premium Charge: "Premium Charge: 10 Pay - 65%/5%/5%/5%/5%"
    Policy Fee: "(starting Y2) 10 Pay -40%/20%/5%/5%"
    Premium Holiday Charge: "None"
    Premium Extension Bonus: "n/a"
    Surrender Charge: ""

  Sunlife Bright:
    Product Name: "Sun MaxiLink Bright"
    Product Type: "Limited Pay VUL"
    Premium Payment Period: "5"
    Currency: "PHP"
    Minimum Premium: "PHP30.5k"
    Minimum SA: "PHP200k"
    SA Flexibility: "No"
    Issue Age: "0 to 60 years old"
    Insurance Coverage: "Up to age 88"
    Death Benefit: "AV + SA"
    Underwriting: "Full Underwriting"
    Mandatory Riders: "None"
    Optional Riders: "ADB, TDB, WPD, WPDD, HIB, CIB"
    Commission: ""
    Premium Charge: "Premium Charge: 5 Pay - 30%/5%/5%"
    Policy Fee: "(starting Y2) 5 Pay - 15%/5%/5%/5%"
    Premium Holiday Charge: "None"
    Premium Extension Bonus: "n/a"
    Surrender Charge: ""

  Manulife:
    Product Name: "FutureBoost"
    Product Type: "Limited Pay VUL"
    Premium Payment Period: "5, 10"
    Currency: "PHP and USD"
    Minimum Premium: "PHP 50k"
    Minimum SA: "PHP750k"
    SA Flexibility: "Yes, with SA Multiplier"
    Issue Age: "0 to 70 years old"
    Insurance Coverage: "Up to age 99"
    Death Benefit: "Higher of (SA, AV)"
    Underwriting: "Full Underwriting"
    Mandatory Riders: "Accident. WP"
    Optional Riders: "Term, HIB, CI"
    Commission: ""
    Premium Charge: "n/a"
    Policy Fee: "Monthly load 5 pay: Var 1 - 75%/30%, Var 2 - 65%/30%, Var 3 - 30%/10%, 10 pay: Var 1 - 80%/75%/25%, Var 2 - 80%/60%/20%, Var 3 - 65%/25%, 100/month"
    Premium Holiday Charge: "None"
    Premium Extension Bonus: "2%"
    Surrender Charge: "100%/90%/80%/70%/60%/50%/40%/30%"

  </productinformation>
  
"""

# Product Knowlegde
product_knowledge = {
        "1a": "Did_AGENT use any wrong insurance terms (reverse)",
        "1b": "Did_AGENT_explain_the_value_of_insurance_in_mitigating_risk",
        "1c": "Did AGENT explain coverage of the policy?",
        "1d": "Did AGENT explain features of the policy?",
        "1e": "Did AGENT give any wrong response regarding policy exclusions? (reverse)",
        "1f": "Did AGENT compare product features with those from competitors when necessary?",
        "1g": "Did AGENT explain how the product fits CUSTOMER's unique situation?",
        "1h": "Did AGENT give an example or analogy of how the policy is in line with current social/economic/market situation?",
}

relationship_management = {
    "2a": "Did AGENT open the conversation with professional and friendly greeting?",
    "2b": "Did AGENT initial small talk or show genuine interest in CUSTOMER to build rapport?",
    "2c": "Did AGENT ask questions to elicit sharing about CUSTOMER background?",
    "2d": "Did AGENT ask questions to elicit sharing about CUSTOMER needs?",
    "2e": "Did AGENT do active listening by asking follow-up question about CUSTOMER's background?",
    "2f": "Did AGENT do active listening by paraphrasing or summarizing what CUSTOMER shared?",
    "2g": "Did AGENT demonstrate personalization by referencing customer-specific information previously shared?",
    "2h": "Did AGENT mention that there will be future follow-up or check-in?",
    "2i": "Did AGENT make explicit commitments to service quality and availability for future assistance?",
    "2j": "Did AGENT give an example of how AGENT helped another customer solve a problem in the past?",
    "2k": "Did AGENT explain how CUSTOMER can get help when needed?",
    "2l": "Did AGENT actively ask for feedback?",
    "2m": "Did AGENT respond constructively to feedback?",
}

sale_skills = {
        "3a": "Did AGENT explain USP (Unique Selling Point) of policy compared to competitor?",
        "3b": "Did AGENT link policy benefits with CUSTOMER's particular situation?",
        "3c": "Did AGENT acknowledge all CUSTOMER objections and concerns with empathy?",
        "3d": "Did AGENT propose a solution or alternative to all CUSTOMER objections and concerns?",
        "3e": "Did AGENT check CUSTOMER readiness to buy before the conversation ended?",
        "3f": "Did AGENT summarize key benefits linked to client needs before the conversation ended?",
        "3g": "Did AGENT state the next steps after this conversation?",
        "3h": "Did AGENT ask questions to assess potential to up-sell or cross-sell?",
}

communication_skills = {
        "4a": "Did AGENT use examples or analogies to explain complex insurance terms?",
        "4b": "Did AGENT ask at least one question to check CUSTOMER's understanding?",
        "4c": "Throughout the whole transcript, did AGENT keep explanation succinct without unnecessary details?",
        "4d": "Did AGENT emphasize key points without unnecessary repetition?",
        "4e": "Did AGENT summarize lengthy discussions?",
        "4f": "Did AGENT adjust communication style and language use to CUSTOMER's style and language?",
        "4g": "Did AGENT quickly address all misunderstandings if any?",
        "4h": "Was AGENT sensitive to cultural norms?",
        "4i": "Did AGENT use words like 'I See' or 'I Understand' to show active listening?",
        "4j": "Did AGENT reflect or paraphrase CUSTOMER statements?",
}

analytical_skills = {
        "5a": "Did AGENT point out issues or gaps in CUSTOMER's current insurance coverage if CUSTOMER has existing insurance coverage?",
        "5b": "Did AGENT ask questions to check CUSTOMER's understanding of insurance coverage?"
}

# Poor not-in-use functions | Not useful for long-form generation
def evalute_agent_knowledge(error_id: str, reason: str, context: str, suggestion: str) -> str:
    """ 
    Parse the evaluation message for the agent's performance. And extract the error id, reason, and suggestion. 
    Args: 
        error_id (str): The error id for the specific error. For instance 1a, 1b, 1c, etc. According to provided information. 
        reason (str): Rationale for the judgement. Why we believe agent shows weekness in product knowledge.
        context (str): Specific context of the error, what does the agent said that is wrong or shows weekness in product knowledge.
        suggestion (str): The suggestion for the agent on how to improve their knowledge.
    """
    issue_dict = {"error_id": "NA", "reason": "NA", "context": "NA", "suggestion": "NA"}
    try:
        issue_dict = {"error_id": error_id, "reason": reason, "context": context, "suggestion": suggestion}
        with open("test.json", "w") as file:
           json.dump(issue_dict, file)
    except:
        issue_dict = {"error_id": "NA", "reason": "NA", "context": "NA", "suggestion": "NA"}







# General One-Shot evaluator 

message_template = """Check the FWD insurance agent's performance in the following transcript, spot the errors and give suggestions on how to improve the agent's performance. Here are some of the potential issues that you can check for \n[POTENTIAL ISSUE] \n{issue_info}\n[TRANSCRIPT]\n{transcript}\n[END] Give your evaluation in format: \n[ATOM] [ATOM DETAIL]\nASSESSMENT: [Yes/No]\nREASON: [Reason for giving a "Yes" or "No"]\nEXTRACT: [Extract one exact line from transcript that gave a "Yes" or "No"]\nIMPROVEMENT: [Provide one example that can be replaced in the conversation transcript that can change a "No" or a "Yes" or a "Yes" to a "No" if atom is reverse]\n[END]"""
message_template_with_knowledge = """Check the FWD insurance agent's performance in the following transcript, spot the errors and give suggestions on how to improve the agent's performance. Here are some relevant product knowledge: \n{knowledge}\n Here are some of the potential issues that you can check for \n[POTENTIAL ISSUE] \n{issue_info}\n[TRANSCRIPT]\n{transcript}\n[END] Give your evaluation in format: \n[ATOM] [ATOM DETAIL]\nASSESSMENT: [Yes/No]\nREASON: [Reason for giving a "Yes" or "No"]\nEXTRACT: [Extract one exact line from transcript that gave a "Yes" or "No"]\nIMPROVEMENT: [Provide one example that can be replaced in the conversation transcript that can change a "No" or a "Yes" or a "Yes" to a "No" if atom is reverse]\n[END]"""

def get_eval_message(issue_dict: dict, transcript: list[str], knowledge: str = "", with_info = False) -> str:
    issue_info = ""
    for key, value in issue_dict.items():
        issue_info += f"- {key}: {value}\n"
    transcript_str = "\n".join(transcript)
    if with_info:
        return message_template_with_knowledge.format(knowledge=knowledge, issue_info=issue_info, transcript=transcript_str)
    else:
        return message_template.format(issue_info=issue_info, transcript=transcript_str)


def direct_parse_judgement(unparsed_judegement: str, issue_dict: dict):
    lines = unparsed_judegement.split("\n")
    error, assessment, reason, extract, improvement = None, None, None, None, None
    judgements = []
    error_ids = list(issue_dict.keys())
    for line in lines:
        for error_id in error_ids:
            if error_id in line:
                error = error_id
                break
        if "ASSESSMENT:" in line and error is not None:
            assessment = "Yes" in line
        if "REASON:" in line and assessment is not None:
            reason = line
        if "EXTRACT:" in line and reason is not None:
            extract = line
        if "IMPROVEMENT:" in line and extract is not None:
            improvement = line
            judgements.append({"error_id": error, "assessment": assessment, "reason": reason, "extract": extract, "improvement": improvement})
            error, assessment, reason, extract, improvement = None, None, None, None, None
    return judgements


def get_attribute_scores(file_name):

    file_path = f"transcripts/feedback/{file_name}.json"
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Initialize a dictionary to store the scores
    attribute_scores = {}

    # Iterate through each error entry in the data
    for entry in data:
        # Extract the error_id and assessment
        error_id = entry["error_id"]
        assessment = entry["assessment"]
        # Extract the attribute number (e.g., '1' from '1a')
        attribute_number = error_id[:-1]
        # Convert assessment to integer (True=1, False=0)
        assessment_score = int(assessment)
        # If the attribute number is already in the dictionary, update it
        if attribute_number in attribute_scores:
            attribute_scores[attribute_number].append(assessment_score)
        else:
            # Otherwise, initialize it in the dictionary
            attribute_scores[attribute_number] = [assessment_score]

    return attribute_scores


def store_judgement(judgements: list, file_name: str):

    file_path =  f"transcripts/feedback/{file_name}.json"
    # Load existing data from the file
    existing_data = []
    try:
        with open(file_path, 'r') as file:
            existing_data = json.load(file)
    except FileNotFoundError:
        print("File not found. Creating a new file.")
        existing_data = []

    # Append new judgements to the existing data
    existing_data.extend(judgements)

    # Write the updated data back to the file
    with open(file_path, 'w') as file:
        json.dump(existing_data, file, indent=4)


def map_score(sub_scores):
    hits = sum(sub_scores)
    max = len(sub_scores)
    if hits == 0:
        return 0.33
    if hits == 1: 
        return 0.6
    if hits == 2:
        return 0.8
    if hits == 3 and hits < max:
        return 0.9
    if hits == 4 and hits < max:
        return 0.95
    else:
        return 1
    
def process_ensemble_score(attribute_scores):
    ensemble_score = {}
    for attribute, sub_scores in attribute_scores.items():
        ensemble_score[attribute] = map_score(sub_scores)
    return ensemble_score


def map_id_to_info(id):
    if id.startswith("1"):
        info = product_knowledge[id]
        attribute = "Product and Industry Knowledge"
    elif id.startswith("2"):
        info = relationship_management[id]
        attribute = "Customer Relationship Management"
    elif id.startswith("3"):
        info = sale_skills[id]
        attribute = "Negotiation and Sales Skills"
    elif id.startswith("4"):
        info = communication_skills[id]
        attribute = "Communication Skills"
    elif id.startswith("5"):
        info = analytical_skills[id]
        attribute = "Analytical Skills"
    return info, attribute


def map_id_to_attribute(id):
    if id.startswith("1"):
        return "Product and Industry Knowledge"
    elif id.startswith("2"):
        return "Customer Relationship Management"
    elif id.startswith("3"):
        return "Negotiation and Sales Skills"
    elif id.startswith("4"):
        return "Communication Skills"
    elif id.startswith("5"):
        return "Analytical Skills"
    

FEEDBACK_REPORT_TEMPLATE = """Based on the detailed evaluation results and scores for each attribute of the FWD insurance sales agent's performance, please provide an evaluation and feedback report. The report should detail the performance in each category, the percentage score, and suggestions for improvement.

[DETAILED EVALUATION]
{eval_str}
[END]

[SCORES]
{scores_str}
[END]

[EXAMPLE REPORT FORMAT]
Category: Product and Industry Knowledge
Percentage: [Percentage from scores_str]
Feedback: [Generated from eval_str]
Improvement: [Generated from eval_str]

Category: Customer Relationship Management
Percentage: [Percentage from scores_str]
Feedback: [Generated from eval_str]
Improvement: [Generated from eval_str]

Category: Negotiation and Sales Skills
Percentage: [Percentage from scores_str]
Feedback: [Generated from eval_str]
Improvement: [Generated from eval_str]


Category: Communication Skills
Percentage: [Percentage from scores_str]
Feedback: [Generated from eval_str]
Improvement: [Generated from eval_str]

Category: Analytical Skills
Percentage: [Percentage from scores_str]
Feedback: [Generated from eval_str]
Improvement: [Generated from eval_str]
[END]
"""



def generate_report(assistant, file_name):

    # Get Scores for Each Attribute & Ensemble the Scores
    attribute_scores = get_attribute_scores(file_name)
    ensemble_score = process_ensemble_score(attribute_scores)

    # Convert the json file into a string, and ask LLM to compress the evaluation result into a proper report 
    file_path = f"transcripts/feedback/{file_name}.json"
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)


    # Prepare Evaluation String
    eval_str = ""
    for p in data:
        info, attribute = map_id_to_info(p["error_id"])
        p_str = f"""\n-- {attribute}: {info}\n[Assessment] {p['assessment']} \n {p['reason']} \n {p['extract']} \n {p['improvement']}"""
        eval_str += p_str
    eval_str = eval_str.strip()
    print(eval_str)

    # Prepare Score String
    scores_str = ""
    for k, v in ensemble_score.items():
        attribute = map_id_to_attribute(str(k))
        score = int(v * 100)
        s_str = f"\nOn {attribute}, the agent has a score of {score}/100."
        scores_str += s_str
    scores_str = scores_str.strip()

    feedback_message = FEEDBACK_REPORT_TEMPLATE.format(eval_str=eval_str, scores_str=scores_str)
    assistant.print_response(feedback_message, markdown=True, stream=False)
    report = assistant.memory.chat_history[-1].content
    file_path = f"transcripts/feedback/{file_name}_feedback_report.txt"
    with open(file_path, "w") as file:
        file.write(report)
    return report
    
