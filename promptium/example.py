
"""
Decorator way of calling ChatGPT.

Create the file /keys/openai with exactly two lines:
1. openai organization id
2. openai api key
(get these in your openai user profile page)

MAKE SURE YOUR KEY DOESNT GET PUSHED TO GITHUB
"""


from promptium.prompt import prompt
import promptium.parse as parse

@prompt
def list_some_scenarios(n, generated=None):
    """
    List {n} diverse examples of everyday tasks that require talking to another person. Format each list item like:

    N. <Role of person 1> talks to <role of person 2> in order to <task goal>


    """
    scenarios = parse.parse(generated, parse.list_items)
    return scenarios


if __name__ == '__main__':

    dialogue_tasks_gpt_made_up = list_some_scenarios(3)
    print(dialogue_tasks_gpt_made_up)







    """
    By default, all calls are cached into a folder called 'llm_cache' in the current directory,
    and all errors are caught and ignored with retrying (using exponential backoff waiting).
    
    You can change these behaviors specifying kwargs when calling your decorated function like below. 
    """
    list_some_scenarios(
        5,
        log=print, # to print the prompt and response as you go
        gen_recache=True, # to invalidate the cache
        cache_folder='my_cache_folder', # to use a different cache folder
        debug=True, # to crash on errors instead of continuing
    )
