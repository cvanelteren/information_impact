import pickle, click

@click.command()
@click.option('--worker_file', help = "pickle file containing worker")
def read_worker_file(worker_file: str):
    with open(worker_file, 'rb') as f:
        worker = pickle.load(f)
    worker.run()

if __name__ == "__main__":
    read_worker_file()
    
