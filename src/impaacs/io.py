# src/impaacs/io.py

def save_fig(fig, path: str) -> None:
    fig.savefig(path, bbox_inches='tight')

def export_results(data, outpath: str) -> None:
    # e.g., write CSV or NetCDF
    pass