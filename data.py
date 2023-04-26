def convert_token(token: str) -> str:
    """Convert PTB tokens to normal tokens."""
    if token.lower() == "-lrb-":
        return "("
    elif token.lower() == "-rrb-":
        return ")"
    elif token.lower() == "-lsb-":
        return "["
    elif token.lower() == "-rsb-":
        return "]"
    elif token.lower() == "-lcb-":
        return "{"
    elif token.lower() == "-rcb-":
        return "}"
    return token 


def get_labels(labels_file: str) -> List[str]:
    """Get a list of relation labels from a .txt file."""
    with open(labels_file) as input_file:
        labels = [label.strip() for label in input_file]
    return labels
