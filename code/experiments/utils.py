def pretty_print_dict(data):
    max_len_key = max(len(k) for k in data.keys())
    lines = []
    for k, v in data.items():
        lines.append(f"{k:<{max_len_key}} : {v}")
    print("=" * len(max(lines, key=len)))
    for line in lines:
        print(line)
    print("=" * len(max(lines, key=len)))