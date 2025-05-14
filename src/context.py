import requests

def get_context_from_wikipedia(subject_entity):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{subject_entity}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("extract", "")  # Returns a short description of the entity
    return ""

def get_context_by_qid(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        description = data["entities"][qid]["descriptions"].get("en", {}).get("value", "")
        return description
    return ""

def get_wikipedia_title(qid):
    """
    Retrieve the title of the English Wikipedia page linked to a Wikidata entity by QID.
    """
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        try:
            title = data["entities"][qid]["sitelinks"]["enwiki"]["title"]
            return title
        except KeyError:
            return "No English Wikipedia page found for this entity."
    else:
        return "Failed to retrieve data from Wikidata."

def main():
    qid = "Q4123315"  
    title = get_wikipedia_title(qid)
    print("English Wikipedia Title:", title)
    context = get_context_from_wikipedia(title)
    print(context) 


if __name__ == "__main__":
    main()
