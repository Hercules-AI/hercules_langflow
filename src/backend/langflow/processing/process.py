from pathlib import Path
from langchain.schema import AgentAction
import json
from langflow.interface.run import (
    build_sorted_vertices_with_caching,
    get_memory_key,
    update_memory_keys,
)
from langflow.utils.logger import logger
from langflow.graph import Graph
from langchain.chains.base import Chain
from langchain.vectorstores.base import VectorStore
from typing import Any, Dict, List, Optional, Tuple, Union


def fix_memory_inputs(langchain_object):
    """
    Given a LangChain object, this function checks if it has a memory attribute and if that memory key exists in the
    object's input variables. If so, it does nothing. Otherwise, it gets a possible new memory key using the
    get_memory_key function and updates the memory keys using the update_memory_keys function.
    """
    if not hasattr(langchain_object, "memory") or langchain_object.memory is None:
        return
    try:
        if (
                hasattr(langchain_object.memory, "memory_key")
                and langchain_object.memory.memory_key in langchain_object.input_variables
        ):
            return
    except AttributeError:
        input_variables = (
            langchain_object.prompt.input_variables
            if hasattr(langchain_object, "prompt")
            else langchain_object.input_keys
        )
        if langchain_object.memory.memory_key in input_variables:
            return

    possible_new_mem_key = get_memory_key(langchain_object)
    if possible_new_mem_key is not None:
        update_memory_keys(langchain_object, possible_new_mem_key)


def format_actions(actions: List[Tuple[AgentAction, str]]) -> str:
    """Format a list of (AgentAction, answer) tuples into a string."""
    output = []
    for action, answer in actions:
        log = action.log
        tool = action.tool
        tool_input = action.tool_input
        output.append(f"Log: {log}")
        if "Action" not in log and "Action Input" not in log:
            output.append(f"Tool: {tool}")
            output.append(f"Tool Input: {tool_input}")
        output.append(f"Answer: {answer}")
        output.append("")  # Add a blank line
    return "\n".join(output)


def get_result_and_thought(langchain_object: Any, inputs: dict):
    """Get result and thought from extracted json"""
    try:
        if hasattr(langchain_object, "verbose"):
            langchain_object.verbose = True

        if hasattr(langchain_object, "return_intermediate_steps"):
            langchain_object.return_intermediate_steps = True

        fix_memory_inputs(langchain_object)

        try:
            output = langchain_object(inputs, return_only_outputs=True)
        except ValueError as exc:
            # make the error message more informative
            logger.debug(f"Error: {str(exc)}")
            output = langchain_object.run(inputs)

    except Exception as exc:
        raise ValueError(f"Error: {str(exc)}") from exc
    return output


def get_input_str_if_only_one_input(inputs: dict) -> Optional[str]:
    """Get input string if only one input is provided"""
    return list(inputs.values())[0] if len(inputs) == 1 else None


def process_graph_cached(data_graph: Dict[str, Any], inputs: Optional[dict] = None):
    """
    Process graph by extracting input variables and replacing ZeroShotPrompt
    with PromptTemplate,then run the graph and return the result and thought.
    """
    # Load langchain object
    langchain_object, artifacts = build_sorted_vertices_with_caching(data_graph)
    logger.debug("Loaded LangChain object")
    if inputs is None:
        inputs = {}

    # Add artifacts to inputs
    # artifacts can be documents loaded when building
    # the flow
    for (
            key,
            value,
    ) in artifacts.items():
        if key not in inputs or not inputs[key]:
            inputs[key] = value

    if langchain_object is None:
        # Raise user facing error
        raise ValueError(
            "There was an error loading the langchain_object. Please, check all the nodes and try again."
        )

    # Generate result and thought
    if isinstance(langchain_object, Chain):
        if inputs is None:
            raise ValueError("Inputs must be provided for a Chain")
        logger.debug("Generating result and thought")
        result = get_result_and_thought(langchain_object, inputs)
        logger.debug("Generated result and thought")
    elif isinstance(langchain_object, VectorStore):
        result = langchain_object.search(**inputs)
    else:
        raise ValueError(
            f"Unknown langchain_object type: {type(langchain_object).__name__}"
        )
    return result


def load_flow_from_json(
        flow: Union[Path, str, dict], tweaks: Optional[dict] = None, build=True
):
    """
    Load flow from a JSON file or a JSON object.

    :param flow: JSON file path or JSON object
    :param tweaks: Optional tweaks to be processed
    :param build: If True, build the graph, otherwise return the graph object
    :return: Langchain object or Graph object depending on the build parameter
    """
    # If input is a file path, load JSON from the file
    if isinstance(flow, (str, Path)):
        with open(flow, "r", encoding="utf-8") as f:
            flow_graph = json.load(f)
    # If input is a dictionary, assume it's a JSON object
    elif isinstance(flow, dict):
        flow_graph = flow
    else:
        raise TypeError(
            "Input must be either a file path (str) or a JSON object (dict)"
        )

    graph_data = flow_graph["data"]
    if tweaks is not None:
        graph_data = process_tweaks(graph_data, tweaks)
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    graph = Graph(nodes, edges)

    if build:
        langchain_object = graph.build()

        if hasattr(langchain_object, "verbose"):
            langchain_object.verbose = True

        if hasattr(langchain_object, "return_intermediate_steps"):
            # Deactivating until we have a frontend solution
            # to display intermediate steps
            langchain_object.return_intermediate_steps = False

        fix_memory_inputs(langchain_object)
        return langchain_object

    return graph


def validate_input(
        graph_data: Dict[str, Any], tweaks: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if not isinstance(graph_data, dict) or not isinstance(tweaks, dict):
        raise ValueError("graph_data and tweaks should be dictionaries")

    nodes = graph_data.get("data", {}).get("nodes") or graph_data.get("nodes")

    if not isinstance(nodes, list):
        raise ValueError(
            "graph_data should contain a list of nodes under 'data' key or directly under 'nodes' key"
        )

    return nodes


def apply_tweaks(node: Dict[str, Any], node_tweaks: Dict[str, Any]) -> None:
    template_data = node.get("data", {}).get("node", {}).get("template")

    if not isinstance(template_data, dict):
        logger.warning(
            f"Template data for node {node.get('id')} should be a dictionary"
        )
        return

    for tweak_name, tweak_value in node_tweaks.items():
        if tweak_name and tweak_value and tweak_name in template_data:
            key = tweak_name if tweak_name == "file_path" else "value"
            template_data[tweak_name][key] = tweak_value


def process_tweaks(
        graph_data: Dict[str, Any], tweaks: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    This function is used to tweak the graph data using the node id and the tweaks dict.

    :param graph_data: The dictionary containing the graph data. It must contain a 'data' key with
                       'nodes' as its child or directly contain 'nodes' key. Each node should have an 'id' and 'data'.
    :param tweaks: A dictionary where the key is the node id and the value is a dictionary of the tweaks.
                   The inner dictionary contains the name of a certain parameter as the key and the value to be tweaked.

    :return: The modified graph_data dictionary.

    :raises ValueError: If the input is not in the expected format.
    """
    nodes = validate_input(graph_data, tweaks)

    for node in nodes:
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            node_id = node["id"]
            if node_tweaks := tweaks.get(node_id):
                apply_tweaks(node, node_tweaks)
        else:
            logger.warning(
                "Each node should be a dictionary with an 'id' key of type str"
            )

    return graph_data


if __name__ == "__main__":
    json_file = "/home/kantharaju/Projects/hercules_langflow/flows/vectore_store_existing_index.json"
    graph_obj = load_flow_from_json(json_file)
    response = (graph_obj("""Wht is the revenue of tesla in 2022?
    
    Context: 
    My passion for quality and clarity drives me to extract valuable insights from vast datasets, while my dedication to staying at the forefront of generative AI advancements allows me to push the boundaries of what's possible. I take pride in leading data initiatives, guiding and inspiring data teams to achieve outstanding results.

One of my key strengths lies in effectively communicating complex findings to stakeholders, bridging the gap between technical intricacies and actionable business strategies. My ability to convey the power of generative AI in transforming data into meaningful outcomes is instrumental in driving organizational growth and success.

With a relentless pursuit of excellence and a focus on delivering impactful solutions, I am committed to making a lasting impact in the realm of data science and generative AI.




User
keep the lenght to the original summary
ChatGPT
As a hybrid data scientist/engineer with over 11 years of experience, I possess unique skills in leveraging software engineering, machine learning, and cloud platforms to build robust data solutions. With a strong dedication to quality and clarity, I am driven by data and focused on delivering results. I excel at leading data initiatives, mentoring data teams, and effectively communicating complex findings to stakeholders. Additionally, I bring expertise in generative AI, empowering me to harness cutting-edge techniques for problem-solving and making a lasting impact in the realm of data science and generative AI.




User
Additionally, I bring expertise in developing generative AI applications. 
ChatGPT
As a hybrid data scientist/engineer with over 11 years of experience, I possess unique skills in leveraging software engineering, machine learning, and cloud platforms to build robust data solutions. With a strong dedication to quality and clarity, I am driven by data and focused on delivering results. I excel at leading data initiatives, mentoring data teams, and effectively communicating complex findings to stakeholders. Additionally, I bring expertise in developing generative AI applications, empowering me to harness cutting-edge techniques for problem-solving and making a lasting impact in the realm of data science and generative AI.




User
Write me a data science portfolio on the project of identifying use cases and industries from customer reviews using Bertopic, transformers, sentence transformers, pytorch ext
ChatGPT
Data Science Portfolio: Identifying Use Cases and Industries from Customer Reviews using Bertopic, Transformers, Sentence Transformers, and PyTorch

Project Overview:
In this data science project, I tackled the challenging task of automatically identifying use cases and industries from customer reviews. The goal was to extract valuable insights from unstructured text data and provide meaningful categorization to help businesses understand customer sentiments and preferences effectively.

Tools and Technologies Used:

Bertopic: A powerful topic modeling library based on BERT embeddings, used for identifying themes and topics in the customer reviews.
Transformers: A popular library for Natural Language Processing (NLP) tasks, employed for fine-tuning BERT-based models to improve topic classification accuracy.
Sentence Transformers: A library for computing dense vector representations of sentences, utilized for transforming text data into numeric embeddings for machine learning models.
PyTorch: An open-source deep learning framework, used for training and fine-tuning NLP models.
Project Steps:

Data Collection and Preprocessing:

Gathered a diverse dataset of customer reviews from various sources, such as online reviews, social media platforms, and survey responses.
Performed data preprocessing, including text cleaning, tokenization, and removal of stop words and special characters.
Bertopic Topic Modeling:

Applied Bertopic to extract latent topics and themes from the customer reviews.
Visualized the discovered topics using interactive visualizations to gain a better understanding of the underlying patterns.
Fine-tuning BERT for Topic Classification:

Utilized the Transformers library to fine-tune a pre-trained BERT model on the annotated data, where each review was labeled with its corresponding use case and industry.
Employed transfer learning to leverage the knowledge learned by BERT on a massive corpus to improve topic classification performance.
Sentence Embeddings with Sentence Transformers:

Utilized Sentence Transformers to transform the preprocessed text data into fixed-length dense vector representations.
Explored various sentence embedding models to find the most suitable one for the classification task.
Model Evaluation and Validation:

Split the dataset into training and testing sets to evaluate the model's performance on unseen data.
Utilized metrics such as accuracy, precision, recall, and F1-score to assess the model's effectiveness.
Deployment and API Integration:

Deployed the trained model as an API service to make real-time predictions on new customer reviews.
Integrated the API into a user-friendly web application to allow businesses to input their customer reviews and receive immediate insights on use cases and industries.
Results and Impact:

The Bertopic-based topic modeling provided valuable insights into the main themes and topics present in the customer reviews, enabling businesses to prioritize areas of improvement.
The fine-tuned BERT model achieved high accuracy in classifying use cases and industries from customer reviews, enhancing the understanding of customer preferences and pain points.
The deployed API and web application facilitated seamless access to the model's predictions, empowering businesses to make data-driven decisions based on customer feedback.
Conclusion:
This data science project demonstrated the powerful capabilities of leveraging state-of-the-art NLP techniques, such as Bertopic, Transformers, Sentence Transformers, and PyTorch, to extract meaningful insights from unstructured text data. The use of deep learning models and topic modeling algorithms enabled businesses to gain a deeper understanding of their customers, identify emerging trends, and make informed decisions to drive success and growth.
    """))
    print(response)
