from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

from search_utils import get_top_results, tfidf_index, idf_dict, doc_vectors

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, query: str = Form(...)):
    results = get_top_results(query, tfidf_index, idf_dict, doc_vectors)
    return templates.TemplateResponse("results.html", {
        "request": request,
        "query": query,
        "results": results
    })

if __name__ == "__main__":
    uvicorn.run("web_search:app", host="0.0.0.0", port=8000, reload=True)
