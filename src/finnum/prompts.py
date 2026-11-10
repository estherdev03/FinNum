def prompt_formatter(
    pre_text: list[str] = [],
    table: list[list[str]] = [],
    post_text: list[str] = [],
    question: str = "",
):
    """
    Construct the prompt containing context and question
    to feed into the model
    Args:
    - pre_text: texts before the table
    - table: table with numbers
    - post_text: texts after the table
    - question: question relate to the provided texts and table
    """

    # Processing pre_text list
    processed_pre_text = ""
    for idx, sentence in enumerate(pre_text):
        if sentence.strip() != "." and sentence.strip() != "":
            processed_pre_text += sentence
            processed_pre_text += "\n" if idx != len(pre_text) - 1 else ""

    # Processing post_text list
    processed_post_text = ""
    for idx, sentence in enumerate(post_text):
        if sentence.strip() != "." and sentence.strip() != "":
            processed_post_text += sentence
            processed_post_text += "\n" if idx != len(post_text) - 1 else ""

    # Processing table
    processed_table = ""
    for idx, row in enumerate(table):
        processed_table += " | ".join(row)
        processed_table += "\n" if idx != len(table) - 1 else ""

    prompt = f"""
    Text before table:
    {processed_pre_text}

    Table:
    {processed_table}

    Text after table:
    {processed_post_text}

    Question: {question}

    Reasoning:
    """
    return prompt
