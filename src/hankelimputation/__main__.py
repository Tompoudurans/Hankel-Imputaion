from .filling import *
import click

@click.command()
@click.option("--file",help="state filename")
@click.option("--batch",help="state batchsize",type="int",default=0)

def main(batch,file):
    data = pd.read_csv(file,skip_blank_lines=False)
    if batch == 0:
        fullfilling(data,file)
    else:
        batchfilling(data,file,batch)


if __name__ == "__main__":
    main()