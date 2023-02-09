from .filling import *
import click


@click.command()
@click.option("--file", help="state filename")
@click.option("--batch", help="state batchsize", type=int, default=0)
@click.option(
    "--e", help="determises how close the model is to data", type=float, default=0.1
)
@click.option(
    "--removezero", help="do you want to remove collaples?", type=bool, default=False
)
@click.option(
    "--maxtime", help="how long before the algorithm time out", type=int, default=50000
)
def main(batch, file, e, removezero, maxtime):
    """
    read a file and fill using the hankel imputaion method
    """
    data = pd.read_csv(file, skip_blank_lines=False)
    filled = processing(data, batch, e, maxtime)
    if removezero:
        filled = refillzero(filled)
    filled.to_csv(file.split(".")[0] + "_filled.csv", index=False)


if __name__ == "__main__":
    main()
