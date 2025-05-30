# marimo-docker

## Setup

1. copy the `.marimo.toml.example` file to `.marimo.toml` and place it in the directory of your choice.
2. update the `docker-compose.prod.yml` file and change the bind mount to the path of your `.marimo.toml` config from step 1. (Note: the current configured location in the docker-compose file is `/opt/docker/marimo/.marimo.toml`)
3. adjust the `.marimo.toml` file with the configuration of your choice. It will now persist through redeploys.
   
