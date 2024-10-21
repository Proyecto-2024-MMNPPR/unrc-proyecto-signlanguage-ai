#!/bin/bash

AI_DIR="server/ai"
AI_LOGS_DIR="$AI_DIR/.logs"

function help() {
    local command=$1
    local subcommand=$2


    if [ -z "$command" ]; then
        echo "Usage: $0 [command] [subcommand]"
        echo "Commands:"
        echo "  help, --help, -h    Show help"
        echo
        echo "  dep                 Dependency commands"
        echo "      install         Install dependencies"
        echo
        echo "  ai                  AI commands"
        echo "      capture         Capture  the samples for training"
        echo "      train           Train the model"
        echo "      run             Run the AI"
        echo
        echo "Examples:"
        echo "  $0 dep install"
        echo "  $0 ai capture"
        echo
        echo "For more information on a command, run '$0 help [command]'."
    else
        case "$command" in
            help)
                help $subcommand
            ;;

            dep)
                echo "usage: $0 dep [install]"
                echo "    dep install          Install dependencies"
            ;;

            ai)
                echo "usage: $0 ai [capture|train|run]"
                echo "    ai capture           Capture samples"
                echo "    ai train             Train the model"
                echo "    ai run               Run the AI"
            ;;

            *)
                echo "    Invalid command. Use 'dep' or 'ai'."
            ;;
        esac
    fi
}

function touch_log() {
    mkdir -p $AI_LOGS_DIR
    touch $AI_LOGS_DIR/$1.log
    "$AI_LOGS_DIR/$1.log"
}

function install_dependencies() {
    echo "Installing dependencies..."

    local errors=0
    echo "Installing AI dependencies..."
    pip install -r ./server/ai/requirements.txt
    errors=$((errors + $?))

    # echo "Installing Web API dependencies..."
    # npm install
    # errors=$((errors + $?))

    echo
    [ $errors -eq 0 ] \
        && echo "Dependencies installed successfully." \
        || echo "There were errors installing dependencies."
}

function capture() {
    local log_file=$(touch_log capture)
    python3 server/ai/capture_samples.py \
    && echo ; echo "Samples captured successfully."
}

function train() {
    local log_file=$(touch_log train)
    python3 server/ai/create_dataset.py \
    && python3 server/ai/train_model.py \
    && echo ; echo "Model trained successfully."
}

function interpret() {
    local log_file=$(touch_log run)
    python3 server/ai/interpret_signs.py
}

case "$1" in
    dep)
        if [ "$2" == "install" ]; then
            install_dependencies
        else
            echo "Invalid subcommand '$2'."
            help dep
        fi
        ;;
    ai)
        case "$2" in
            capture)
                capture
                ;;
            train)
                train
                ;;
            run)
                interpret
                ;;
            *)
                echo "Invalid subcommand '$2'."
                help ai
                ;;
        esac
        ;;
    *)
        echo "Invalid command '$1'."
        help
        ;;
esac
