# cog-bias-med-LLMS
Addressing common clinical biases in medical language models


## Results so far

Tested on 500 medQA test set data points
#### gpt-3.5-turbo-0613
No bias: 0.51

Recency bias: 0.36

#### gpt-4-0613
No bias: 

Recency bias: 0.724

## API keys
Originally, these were hard-coded. Since we're using GitHub and may publicly release this code, this is probably not the best idea. To ameriorate this, we should use environmental variables set outside of the repo, and then call them via `os.environ.get(OPENAI_API_KEY)`. On Mac, persistent environment variables can be set in `~/.zshrc` (e.g., by adding export `OPENAI_API_KEY="sk-..."` to the end of the file). Check the slack for CH's API key.
