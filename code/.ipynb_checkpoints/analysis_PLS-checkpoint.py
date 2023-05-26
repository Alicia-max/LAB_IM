from behavPLS import BehavPLS
import pickle
import typer

app = typer.Typer()

@app.command()
def main(name: str):
    print(name)
      
    dataset=BehavPLS(name)
    res = dataset.run_pls()
    
    with open('saved_res.pkl', 'wb') as f:
        pickle.dump(res, f)
    with open('saved_res.pkl', 'rb') as f:
        loaded_dict = pickle.load(f)
        
        
    

    return res 
   

if __name__ == "__main__":
    app()
    
    
