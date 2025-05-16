# fabric-help-bot

## To run

```
source .venv/bin/activate
python fabric_ai_helper.py
```


## config file format


```
[SERVER]
host_url=<server.edu>

[VectorDB]
forum_db_loc = <path/to/db>
kb_db_loc = <path/to/db>
kb_forum_db_loc = <path/to/db>

[API_KEYS]
openai_key = <api key>

[USERS]
<username> = <password>
```


## For notebooks

```
jupyter lab [--ip 0.0.0.0]
```


## Python Version (Tested)
3.12

