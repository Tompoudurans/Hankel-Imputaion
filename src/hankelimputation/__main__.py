from .filling import *
import click


@click.command()
@click.option("--file", help="state filename")
@click.option("--batch", help="state batchsize", type=int, default=0)
@click.option(
    "--epsilon", help="determises how close the model is to data", type=float, default=0.1
)
@click.option(
    "--double", help="use dual-hi?", type=bool, default=True
)
@click.option(
    "--maxtime", help="how long before the algorithm time out", type=int, default=50000
)
@click.option(
    "--indexcol", help="detrmine if got index", type=int, default=0
)
def main(batch, file, epsilon, double, maxtime,indexcol):
    """
    read a file and fill using the hankel imputaion method
    """
    data = pd.read_csv(file, skip_blank_lines=False,index_col=indexcol)
    print("filling",data.isna().sum().sum(),"missing values out of",data.shape[0]*data.shape[1],"datapoints")
    filled = processing(data, batch, epsilon, double=double, verbose=True, max_iters=maxtime)
    filled.to_csv(file.split(".")[0] + "_filled.csv", index=False)


if __name__ == "__main__":
    main()
