## Setup

* Create your virtual environment
    - `conda create -n fact-check python=3.9` or `python3 -m venv .venv`
* Activate environment
    - `conda activate fact-check` or `source .venv/bin/activate`
* Install packages
    - `pip3 install -r requirements.txt`
* Download model into `models/`
* (option 1) Create sample database
    - `cd scripts` then `python3 create_sample_database.py` then `python3 embed_bills.py`
* (option 2) Download Congress DB below and place in `data/`
* (option 3) Scrape it yourself
* Run code
    - Example: `python3 interact.py "John Smith voted against improving our schools" --device=cuda`


## Resources

Congress DB: https://drive.google.com/file/d/1USmzLbW1b04JLTQzq9usL7JHcBMrcPZY/view?usp=sharing

Voting records scraping: https://github.com/unitedstates/congress 
