"""Prompt templates.

The belief-tree and confidence prompts are copied word for word from the official BTProp
repository (github.com/UCSB-NLP-Chang/BTProp, src/prompts.py, Llama variants), so our trees are
built the way the paper built them -- see docs/project-log/03-decisions.md, D29. The repository
has no licence file; the texts are reproduced here with attribution, for research reproduction.

Only DECONTEXTUALIZE is ours: the paper evaluated on a hand-cleaned FELM-Science file (not public)
where every sentence already stands alone; we approximate that step with the LLM.
"""

from __future__ import annotations

# official: src/prompts.py :: llama_confidence_prob
LLAMA_CONFIDENCE = """Are the following statements true or false? For each of the following statements, determine whether it is true or false. Provide a response of 'True' if the statement is correct, or 'False' if the statement is incorrect.

Remember:
    - You **do not** have access to additional information or external data.
    - If verifying the statement requires such external data or context, regard it as false.
    - Directly output "True" or "False" without adding any markers."""


# official: src/prompts.py :: llama_decompose
LLAMA_DECOMPOSE = """**Fact-Checking Claims Extraction:**

**Objective:** Analyze the provided statement to extract and list all distinct factual claims that require verification. Each listed claim should be verifiable and not overlap in content with other claims listed.

**Instructions:**
1. Identify parts of the statement that assert specific, verifiable facts. In every claim, alway use the **full names** when referring to any concept, person, or entity. **Avoiding the use of pronouns or indirect references** that require contextual understanding.

2. List each verifiable claim separately. Ensure that each claim is distinct and there is no overlap in the factual content between any two claims.

3. Exclude any statements that are purely hypothetical, assume theoretical scenarios, or are speculative in nature. These do not contain verifiable factual claims. For example, statements involving assumptions ("Let's assume..."), theoretical discussions ("consider whether..."), or purely speculative scenarios should not be considered as containing verifiable claims.

4. If the statement does not contain any verifiable facts, output the following message: "Claim 1: No check-worthy claims available."

**Output Format**:
Your output should be organized as follows:
Claim 1: <the first claim>
Claim 2: <the second claim>
Claim 3: <the third claim (if necessary)>
...

**Examples:**
Statement: According to recent data, China has surpassed the United States in terms of GDP when measured using Purchasing Power Parity (PPP), and India is projected to overtake China by 2030.
Claim 1: China has surpassed the United States in terms of GDP when measured using Purchasing Power Parity (PPP).
Claim 2: India is projected to overtake China in terms of GDP by 2030."

Statement: The world's largest desert is Antarctica, and it is larger than the Sahara.
Claim 1: The world's largest desert is Antarctica.
Claim 2: Antarctica is larger than the Sahara.

Statement: I think pizza is the best food ever!
Claim 1: No check-worthy claims available.

Statement: Let's assume there is a highest prime number and consider its implications on number theory.
Claim 1: No check-worthy claims available."""


# official: src/prompts.py :: llama_select_strategy
LLAMA_SELECT_STRATEGY = """**Instruction for Choosing the Best Strategy for Premise Generation**

When tasked with generating premises for a given target statement, the choice of strategy—Logical Relationships or Statement Perturbation—should be determined based on the nature of the statement and the desired objectives. Use the following guidelines to route the choice:

### When to Use Logical Relationships
Opt for Logical Relationships when:
- **The Statement is Abstract or Principled**: Ideal for statements that explore broad principles, ethics, or abstract concepts. This method helps in drawing deep logical entailments or contradictions.
- **Complex Relationships or Conditions**: When the statement involves complex logical or conditional relationships, using this method clarifies or challenges these intricacies.
- **Requirement for Detailed Analysis**: For statements needing precise and formal argumentation, especially in academic or technical discussions.

### When to Use Statement Perturbation
Choose Statement Perturbation when:
- **The Statement is Specific and Concrete**: Best for statements with explicit details like scenarios, dates, locations, names, numbers or timestamps. Altering these elements generates varied premises.
- **Exploration of Counterfactuals or Hypotheticals**: Useful for creating imaginative or scenario-based premises by modifying key details to envision different outcomes.
- **Sensitivity to Detail Changes**: When minor modifications in the statement can significantly alter its implications or truth value.

**Output Format:**
Your selection should be one of "Logical Relation", "Statement Perturbation", and "both". Here "both" means both two methods are applicable.

**Examples:**
Target statement: Eating a balanced diet improves overall health.
Output: Logical Relation

Target statement: Countries with higher investment in education consistently rank higher in global innovation indexes
Output: Logical Relation

Target statement: 'War and Peace' was a book written by Leo Tolstoy.
Output: Statement Perturbation

Target statement: Water boils at 100 degrees Celsius at sea level
Output: both

Target statement: Mount Everest is 8,848 meters tall.
Output: Statement Perturbation

Target statement: The bridge will remain intact even if 75 cars drive on it simultaneously if the cars are lightweight
Output: Statement Perturbation

Target statement: Artificial intelligence will replace most human jobs in the future
Output: Logical Relation"""


# official: src/prompts.py :: support_via_explanation
SUPPORT_VIA_EXPLANATION = """**Finding Supportive Premises**

Is the following statement true or false? If it is true, list several supportive premises for it.

**Important Rules:**
1. Each premise should be clearly stated and directly relevant to the target statement. Avoid ambiguity and ensure that the connection to the target statement is evident
2. Do not use pronouns in generated premises. Ensure each premise can be understood clearly without any context. For each generated premise, you should always use the full name of each person, event, object, etc.

**Input/Output Format**:
Your output should be organized as follows.
Judgement: <True or False>
Premise 1: <the first premise>
Premise 2: <the second premise>
...

In contrast, if the statement is false, you simly output:
Judgement: False
Premise 1: No supportive premises applicable.

**Examples:**
Target statement: Renewable energy sources will lead to a decrease in global greenhouse gas emissions.
Judgement: True
Premise 1: Renewable energy sources produce electricity without emitting carbon dioxide.
Premise 2: Increasing the adoption of renewable energy reduces reliance on fossil fuels, which are the primary source of industrial carbon dioxide emissions.

Target statement: Eating carrots improves night vision.
Judgement: False
Premise 1: No supportive premises applicable.

Statement: Historical literacy enhances a society’s ability to make informed decisions.
Judgement: True
Premise 1: Understanding historical events provides context for current issues, enabling citizens to make decisions that consider past outcomes and lessons.
Premise 2: Historical literacy fosters critical thinking skills, which are crucial in analyzing information and making reasoned decisions.
Premise 3: Societies with high levels of historical awareness can recognize and avoid the repetition of past mistakes."""


# official: src/prompts.py :: llama_perturb
LLAMA_PERTURB = """Given a statement, your task is to identify a general wh-question that can be used to check the truthfulness of the statement. This question should directly address the main claim of the statement to confirm or refute it, using a 'wh-' interrogative form (e.g., who, what, where, when, why, how) and should not seek additional detailed information.

**Examples:**
Statement: Bitcoin was created in 2009 by an anonymous entity known as Satoshi Nakamoto.
Question: Who created Bitcoin in 2009?

Statement: The iPhone was first released by Apple in 2007.
Question: When was iPhone first relased by Apple?

Statement: 'Romeo and Juliet' was written by Shakespeare in the late 16th century.
Question: When was 'Romeo and Juliet' was written by Shakespeare?

Statement: "Attention Is All You Need" is a paper written by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
Question: Who was/were the authors of the paper "Attention Is All You Need"?

Statement: The Eiffel Tower is located in Paris
Question: Where is the Eiffel Tower located?

Statement: Albert Einstein was a physicist.
Question: What profession was Albert Einstein?"""


# official: src/prompts.py :: llama_perturb_fillin
LLAMA_PERTURB_FILLIN = """**Background Knowledge**: {background}

Leverage the above provided knowledge and your own knowledge to review the correctness of following statement:

**Statement**: {statement}

Instruction:
    - If the statement is correct, output it unchanged.
    - If the statement is **not mentioned in the background knowledge and its correctness cannot be determined**, you should also directly output the statement **unchanged**.
    - If the statement is wrong, revise only the parts of the statement that are incorrect, to align with the background knowledge. Do not add any additional sentences or details.

**Output Format:**
Format your output as:
Revised Answer: <Display the original statement if it is correct or not mentioned in the background knowledge; display the revised statement if it is inaccurate>"""


DECONTEXTUALIZE = """Below is a question, an AI assistant's answer to it, and one sentence taken from that answer.

Rewrite the sentence so that it can be understood on its own, without the question or the rest of the answer:
    - Replace pronouns and vague references ("it", "they", "this process", "the former") with the full names they refer to.
    - Add only the context needed to make the sentence understandable. Do not add any other facts.
    - Keep the meaning exactly the same, including any mistakes. Do not correct, soften or judge the sentence.
    - If the sentence already stands on its own, output it unchanged.

**Output Format:**
Standalone sentence: <the rewritten sentence>

**Question**: {question}

**Answer**: {answer}

**Sentence**: {sentence}"""


def llama_confidence_messages(statement: str) -> list[dict[str, str]]:
    """The paper's Llama confidence query: instruction, a fixed acknowledgement turn, then the
    statement (official confidence_estimation.py). We read P(True) vs P(False) for the next token."""
    return [
        {"role": "user", "content": LLAMA_CONFIDENCE},
        {
            "role": "assistant",
            "content": "Yes, I understand the task. You can provide the statement"
            " you'd like me to guess the truthfulness.",
        },
        {"role": "user", "content": "Statement: " + statement.strip()},
    ]


def true_false_confidence_prompt(statement: str) -> str:
    """Single-turn variant for API clients that take one prompt string (OpenAIClient)."""
    return LLAMA_CONFIDENCE + "\n\nStatement: " + statement.strip()


def decomposition_prompt(statement: str) -> str:
    return LLAMA_DECOMPOSE + "\n\nStatement: " + statement


def select_strategy_prompt(statement: str) -> str:
    return LLAMA_SELECT_STRATEGY + "\n\nTarget statement: " + statement


def support_premises_prompt(statement: str) -> str:
    return SUPPORT_VIA_EXPLANATION + "\n\nStatement: " + statement


def perturb_question_prompt(statement: str) -> str:
    return LLAMA_PERTURB + "\n\nStatement: " + statement


def perturb_fillin_prompt(background: str, statement: str) -> str:
    return LLAMA_PERTURB_FILLIN.format(background=background, statement=statement)


def decontextualize_prompt(question: str, answer: str, sentence: str) -> str:
    return DECONTEXTUALIZE.format(question=question, answer=answer, sentence=sentence)
