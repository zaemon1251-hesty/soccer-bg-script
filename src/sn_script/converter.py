import sys
import tomllib


def convert(d: dict[str, str]) -> list[str]:
    ll = [
        k + "==" + v.replace("~", "").replace("^", "") # ここはご自由に
        for k, v in d.items()
        if k != "python"
    ]
    return ll


def main(poetry_pyproject: str) -> None:
    with open(poetry_pyproject, "rb") as f:
        d = tomllib.load(f)
        try:
            deps = d["tool"]["poetry"]["dependencies"]
            print("[project]\ndependencies = [")
            [print(f'    "{x}",') for x in convert(deps)]
            print("]")
        except Exception as _:
            pass

        try:
            dev_deps = d["tool"]["poetry"]["group"]["dev"]["dependencies"]
            print("[tool.uv]\ndev-dependencies = [")
            [print(f'    "{x}",') for x in convert(dev_deps)]
            print("]")
        except Exception as _:
            print("## there is no [tool.poetry.group.dev.dependencies]")


if __name__ == "__main__":
    args = sys.argv
    if len(args) != 2:
        sys.exit("bad arguments")

    main(args[1])

