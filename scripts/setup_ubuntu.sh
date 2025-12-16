#!/bin/bash
#
# QualSynth Ubuntu Setup and Experiment Runner
# 
# This script sets up a complete environment for running QualSynth experiments
# on Ubuntu 20.04/22.04/24.04 (including WSL2 on Windows)
#
# Usage:
#   chmod +x setup_ubuntu.sh
#   ./setup_ubuntu.sh              # Full setup + run experiments
#   ./setup_ubuntu.sh --setup-only # Only install dependencies
#   ./setup_ubuntu.sh --run-only   # Only run experiments (assumes setup done)
#   ./setup_ubuntu.sh --quick      # Quick replication with pre-computed results
#
# Requirements:
#   - Ubuntu 20.04+ or compatible Linux distribution
#   - At least 16GB RAM (32GB recommended)
#   - NVIDIA GPU with 8GB+ VRAM (optional, for faster inference)
#   - ~50GB free disk space
#   - Internet connection for downloading models
#


set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
OLLAMA_MODEL="gemma3:12b"  # Gemma 3 12B parameter model
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Determine if we need sudo (check if running as root or if sudo exists)
if [[ $EUID -eq 0 ]]; then
    SUDO=""
elif command -v sudo &> /dev/null; then
    SUDO="sudo"
else
    SUDO=""
    print_warning "Neither root nor sudo available - some installations may fail"
fi

# Parse arguments
SETUP_ONLY=false
RUN_ONLY=false
QUICK_MODE=false
DRY_RUN=false

for arg in "$@"; do
    case $arg in
        --setup-only)
            SETUP_ONLY=true
            ;;
        --run-only)
            RUN_ONLY=true
            ;;
        --quick)
            QUICK_MODE=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --setup-only    Only install dependencies, don't run experiments"
            echo "  --run-only      Only run experiments (assumes setup is complete)"
            echo "  --quick         Quick replication using pre-computed results"
            echo "  --dry-run       Preview what would be done without executing"
            echo "  --help, -h      Show this help message"
            exit 0
            ;;
    esac
done

print_header() {
    echo ""
    echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC} $1"
    echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

check_system() {
    print_header "Checking System Requirements"
    
    # Check OS
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        print_step "OS: $NAME $VERSION_ID"
    else
        print_warning "Could not detect OS version"
    fi
    
    # Check RAM
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    if [[ $TOTAL_RAM -lt 16 ]]; then
        print_warning "RAM: ${TOTAL_RAM}GB (16GB+ recommended)"
    else
        print_step "RAM: ${TOTAL_RAM}GB"
    fi
    
    # Check disk space
    FREE_SPACE=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
    if [[ $FREE_SPACE -lt 50 ]]; then
        print_warning "Free space: ${FREE_SPACE}GB (50GB+ recommended)"
    else
        print_step "Free space: ${FREE_SPACE}GB"
    fi
    
    # Check for NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
        GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
        print_step "GPU: $GPU_NAME ($GPU_MEM)"
        print_step "GPU Utilization: $GPU_UTIL"
        HAS_GPU=true
        
        # Check CUDA version
        if nvidia-smi --query-gpu=driver_version --format=csv,noheader &>/dev/null; then
            DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
            print_step "NVIDIA Driver: $DRIVER_VER"
        fi
    else
        print_warning "No NVIDIA GPU detected (CPU inference will be slower)"
        print_warning "Ollama will run on CPU - expect ~10x slower inference"
        HAS_GPU=false
    fi
    
    echo ""
}

install_system_dependencies() {
    print_header "Installing System Dependencies"
    
    # Update package lists
    print_step "Updating package lists..."
    $SUDO apt-get update -qq
    
    # Install essential packages
    print_step "Installing essential packages..."
    $SUDO apt-get install -y -qq \
        curl \
        wget \
        git \
        build-essential \
        software-properties-common \
        ca-certificates \
        gnupg \
        lsb-release
    
    print_step "System dependencies installed"
}

install_python() {
    print_header "Setting Up Python Environment"
    
    # Try to find an available Python version (3.10, 3.9, 3.8, or system python3)
    PYTHON_CMD=""
    PYTHON_FULL_VERSION=""
    for ver in "3.10" "3.11" "3.9" "3.8" "3"; do
        if command -v python$ver &> /dev/null; then
            PYTHON_CMD="python$ver"
            PYTHON_VERSION="$ver"
            PYTHON_FULL_VERSION=$(python$ver --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
            print_step "Found Python: $PYTHON_CMD ($(python$ver --version 2>&1))"
            break
        fi
    done
    
    if [[ -z "$PYTHON_CMD" ]]; then
        print_step "No Python found, installing Python 3..."
        $SUDO apt-get update -qq
        $SUDO apt-get install -y -qq python3 python3-venv python3-dev python3-pip
        PYTHON_CMD="python3"
        PYTHON_VERSION="3"
        PYTHON_FULL_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    fi
    
    # Get the actual minor version for package names (e.g., 3.8 from Python 3.8.x)
    if [[ -z "$PYTHON_FULL_VERSION" ]]; then
        PYTHON_FULL_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    fi
    
    print_step "Python version detected: $PYTHON_FULL_VERSION"
    
    # Install venv and dev packages for the specific Python version
    print_step "Installing python${PYTHON_FULL_VERSION}-venv and dependencies..."
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq \
        python${PYTHON_FULL_VERSION}-venv \
        python${PYTHON_FULL_VERSION}-dev \
        python3-pip \
        2>/dev/null || \
    $SUDO apt-get install -y -qq \
        python3-venv \
        python3-dev \
        python3-pip
    
    # Verify venv works
    if ! $PYTHON_CMD -m venv --help &> /dev/null; then
        print_error "Failed to install python venv module"
        print_warning "Try manually: apt install python${PYTHON_FULL_VERSION}-venv"
        exit 1
    fi
    
    # Export for use in other functions
    export PYTHON_CMD
    export PYTHON_FULL_VERSION
    
    print_step "Python setup complete: $PYTHON_CMD"
}

setup_virtual_environment() {
    print_header "Setting Up Virtual Environment"
    
    VENV_DIR="$PROJECT_DIR/venv"
    REPLICATION_DIR="$PROJECT_DIR/replication"
    
    # Use PYTHON_CMD if set, otherwise find python
    if [[ -z "$PYTHON_CMD" ]]; then
        for ver in "3.10" "3.11" "3.9" "3.8" "3"; do
            if command -v python$ver &> /dev/null; then
                PYTHON_CMD="python$ver"
                break
            fi
        done
    fi
    
    if [[ -d "$VENV_DIR" ]]; then
        print_step "Virtual environment already exists"
    else
        print_step "Creating virtual environment with $PYTHON_CMD..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Clear pip cache to avoid corrupted packages
    print_step "Clearing pip cache..."
    pip cache purge 2>/dev/null || rm -rf ~/.cache/pip 2>/dev/null || true
    
    # Upgrade pip and setuptools
    print_step "Upgrading pip and setuptools..."
    pip install --upgrade pip setuptools wheel --no-cache-dir -q
    
    # Install Python dependencies from replication folder (Python 3.8 compatible)
    print_step "Installing Python dependencies..."
    pip install --no-cache-dir -r "$REPLICATION_DIR/requirements.txt"
    
    # Install QualSynth package from replication folder (without build isolation to avoid cache issues)
    print_step "Installing QualSynth package..."
    pip install --no-cache-dir --no-build-isolation -e "$REPLICATION_DIR/qualsyn-1.0.0"
    
    print_step "Virtual environment ready"
}

install_ollama() {
    print_header "Installing Ollama"
    
    if command -v ollama &> /dev/null; then
        print_step "Ollama already installed"
        OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "unknown")
        print_step "Version: $OLLAMA_VERSION"
    else
        print_step "Downloading and installing Ollama..."
        curl -fsSL https://ollama.com/install.sh | sh
        print_step "Ollama installed"
    fi
    
    # Start Ollama service
    print_step "Starting Ollama service..."
    if systemctl is-active --quiet ollama 2>/dev/null; then
        print_step "Ollama service already running"
    else
        # Try systemctl first, fall back to manual start
        if command -v systemctl &> /dev/null; then
            $SUDO systemctl start ollama 2>/dev/null || nohup ollama serve > /tmp/ollama.log 2>&1 &
        else
            nohup ollama serve > /tmp/ollama.log 2>&1 &
        fi
        sleep 5
    fi
    
    # Verify Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_step "Ollama server is running"
        
        # Check if Ollama is using GPU
        if [[ "$HAS_GPU" == true ]]; then
            # Check GPU memory usage after Ollama starts
            sleep 2
            GPU_MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
            print_step "GPU memory in use: $GPU_MEM_USED"
            print_step "Ollama will use GPU for inference (faster)"
        else
            print_warning "Ollama running on CPU (slower inference)"
        fi
    else
        print_error "Failed to start Ollama server"
        exit 1
    fi
}

download_model() {
    print_header "Downloading LLM Model"
    
    print_step "Pulling $OLLAMA_MODEL model (this may take a while)..."
    ollama pull $OLLAMA_MODEL
    
    # Verify model
    if ollama list | grep -q "$OLLAMA_MODEL"; then
        print_step "Model $OLLAMA_MODEL ready"
    else
        print_error "Failed to download model"
        exit 1
    fi
}

generate_data_splits() {
    print_header "Generating Data Splits"
    
    cd "$PROJECT_DIR"
    source "$PROJECT_DIR/venv/bin/activate"
    
    print_step "Creating fresh data splits for all datasets..."
    print_step "This ensures compatibility with the installed scikit-learn version"
    
    # Create splits directory if it doesn't exist
    mkdir -p "$PROJECT_DIR/data/splits"
    
    python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/replication/qualsyn-1.0.0')

from qualsyn.data.splitting import create_splits

datasets = ['german_credit', 'breast_cancer', 'pima_diabetes', 'wine_quality', 
            'yeast', 'haberman', 'thyroid', 'htru2']
seeds = [42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789]

print('Generating splits for 8 datasets x 10 seeds = 80 split files')
print('=' * 60)

for dataset in datasets:
    print(f'\\nDataset: {dataset}')
    try:
        create_splits(dataset, seeds=seeds, output_dir='$PROJECT_DIR/data/splits')
        print(f'  ✓ Created 10 splits')
    except Exception as e:
        print(f'  ✗ Error: {e}')
        
print('\\n' + '=' * 60)
print('Split generation complete!')
"
    
    # Verify splits were created
    SPLIT_COUNT=$(find "$PROJECT_DIR/data/splits" -name "*.pkl" 2>/dev/null | wc -l)
    print_step "Generated $SPLIT_COUNT split files"
}

run_quick_replication() {
    print_header "Running Quick Replication (Pre-computed Results)"
    
    cd "$PROJECT_DIR/replication"
    source "$PROJECT_DIR/venv/bin/activate"
    
    python replication.py --quick
    
    print_step "Quick replication complete!"
}

run_experiments() {
    print_header "Running Full Experiments"
    
    cd "$PROJECT_DIR"
    source "$PROJECT_DIR/venv/bin/activate"
    
    # Generate fresh data splits to ensure compatibility with installed sklearn version
    generate_data_splits
    
    # Verify split files exist
    print_step "Verifying data splits..."
    SPLITS_DIR="$PROJECT_DIR/data/splits"
    
    SPLIT_COUNT=$(find "$SPLITS_DIR" -name "*.pkl" 2>/dev/null | wc -l)
    if [[ $SPLIT_COUNT -lt 80 ]]; then
        print_error "Only $SPLIT_COUNT split files found (expected 80)"
        print_error "Split generation may have failed"
        exit 1
    else
        print_step "Found $SPLIT_COUNT split files"
    fi
    
    # Set environment variables for Ollama
    export OPENAI_API_BASE="http://localhost:11434/v1"
    export OPENAI_API_KEY="ollama"  # Ollama accepts any non-empty string
    export OLLAMA_MODEL="$OLLAMA_MODEL"
    
    # Datasets and methods to run
    DATASETS=("german_credit" "breast_cancer" "pima_diabetes" "wine_quality" "yeast" "haberman" "thyroid" "htru2")
    METHODS=("qualsynth" "smote" "ctgan" "tabfairgdt")
    SEEDS=(42 123 456 789 1234 2024 3141 4242 5555 6789)
    
    # Create results directory (in project root, organized by method)
    RESULTS_DIR="$PROJECT_DIR/results/ubuntu_experiments"
    mkdir -p "$RESULTS_DIR"
    
    # Log file
    LOG_FILE="$RESULTS_DIR/experiment_log_$(date +%Y%m%d_%H%M%S).txt"
    
    echo "Experiment Log - $(date)" > "$LOG_FILE"
    echo "==============================" >> "$LOG_FILE"
    
    TOTAL_EXPERIMENTS=$((${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))
    CURRENT=0
    
    print_step "Running $TOTAL_EXPERIMENTS experiments..."
    print_step "Log file: $LOG_FILE"
    echo ""
    
    for dataset in "${DATASETS[@]}"; do
        for method in "${METHODS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                CURRENT=$((CURRENT + 1))
                
                # Check if result already exists (for resuming)
                RESULT_FILE="$RESULTS_DIR/${dataset}_${method}_seed${seed}.json"
                if [[ -f "$RESULT_FILE" ]]; then
                    echo "[$CURRENT/$TOTAL_EXPERIMENTS] Skipping $dataset/$method/seed$seed (already exists)"
                    continue
                fi
                
                echo "[$CURRENT/$TOTAL_EXPERIMENTS] Running: $dataset / $method / seed $seed"
                echo "$(date): $dataset / $method / seed $seed" >> "$LOG_FILE"
                
                # Run experiment
                START_TIME=$(date +%s)
                
                if python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR/replication')

from qualsyn import QualSynthGenerator
from qualsyn.data.splitting import load_split
import json
import traceback

try:
    # Load data (raw format for LLM to understand feature meanings)
    split = load_split('$dataset', seed=$seed, return_raw=True)
    X_train, y_train = split['X_train'], split['y_train']
    X_test, y_test = split['X_test'], split['y_test']
    
    if '$method' == 'qualsynth':
        # Run QualSynth using simple API
        generator = QualSynthGenerator(
            model_name='$OLLAMA_MODEL',
            temperature=0.7,
            max_iterations=20
        )
        X_syn, y_syn = generator.fit_generate(X_train, y_train)
        
        # Save result
        with open('$RESULT_FILE', 'w') as f:
            json.dump({
                'dataset': '$dataset',
                'method': '$method',
                'seed': $seed,
                'n_generated': len(X_syn) if X_syn is not None else 0,
                'validation_rate': generator.validation_rate_,
                'success': True
            }, f, indent=2)
    else:
        # Run baseline methods
        from baselines import SMOTEBaseline, CTGANBaseline, TabFairGDTBaseline
        
        if '$method' == 'smote':
            baseline = SMOTEBaseline(random_state=$seed)
        elif '$method' == 'ctgan':
            baseline = CTGANBaseline(random_state=$seed)
        elif '$method' == 'tabfairgdt':
            baseline = TabFairGDTBaseline(random_state=$seed)
        
        X_res, y_res = baseline.fit_resample(X_train, y_train)
        
        with open('$RESULT_FILE', 'w') as f:
            json.dump({
                'dataset': '$dataset',
                'method': '$method', 
                'seed': $seed,
                'n_generated': len(X_res) - len(X_train),
                'success': True
            }, f, indent=2)
            
    print('SUCCESS')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
    with open('$RESULT_FILE', 'w') as f:
        json.dump({
            'dataset': '$dataset',
            'method': '$method',
            'seed': $seed,
            'success': False,
            'error': str(e)
        }, f, indent=2)
" 2>&1 | tee -a "$LOG_FILE"; then
                    END_TIME=$(date +%s)
                    DURATION=$((END_TIME - START_TIME))
                    echo "  Completed in ${DURATION}s" | tee -a "$LOG_FILE"
                else
                    echo "  Failed" | tee -a "$LOG_FILE"
                fi
                
                echo "" >> "$LOG_FILE"
            done
        done
    done
    
    print_step "All experiments complete!"
    print_step "Results saved to: $RESULTS_DIR"
    print_step "Log file: $LOG_FILE"
}

# Dry run function
run_dry_run() {
    print_header "DRY RUN - Preview of Setup Steps"
    
    echo -e "${YELLOW}This is a preview. No changes will be made.${NC}"
    echo ""
    
    echo "[1/6] System Check"
    echo "   Would check: OS version, RAM, disk space, GPU"
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "   Current OS: $NAME $VERSION_ID"
    fi
    TOTAL_RAM=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "N/A")
    echo "   Current RAM: ${TOTAL_RAM}GB"
    echo ""
    
    echo "[2/6] System Dependencies"
    echo "   Would install via apt-get:"
    echo "   - curl, wget, git, build-essential"
    echo "   - software-properties-common, ca-certificates"
    echo ""
    
    echo "[3/6] Python Setup"
    echo "   Would install: Python $PYTHON_VERSION"
    echo "   Would create venv at: $PROJECT_DIR/venv"
    echo "   Would install packages:"
    if [[ -f "$PROJECT_DIR/requirements.txt" ]]; then
        grep -v "^#" "$PROJECT_DIR/requirements.txt" | grep -v "^$" | head -8 | sed 's/^/     - /'
        echo "     ... (and more)"
    fi
    echo ""
    
    echo "[4/6] Ollama Installation"
    if command -v ollama &> /dev/null; then
        echo "   ✓ Ollama already installed: $(ollama --version 2>/dev/null || echo 'version unknown')"
    else
        echo "   Would download and install Ollama from https://ollama.com"
    fi
    echo ""
    
    echo "[5/6] LLM Model Download"
    echo "   Would download: $OLLAMA_MODEL (~8GB)"
    if command -v ollama &> /dev/null; then
        echo "   Currently available models:"
        ollama list 2>/dev/null | head -5 | sed 's/^/     /' || echo "     (none)"
    fi
    echo ""
    
    echo "[6/6] Experiments"
    echo "   Datasets: german_credit, breast_cancer, pima_diabetes,"
    echo "             wine_quality, yeast, haberman, thyroid, htru2"
    echo "   Methods: qualsynth, smote, ctgan, tabfairgdt"
    echo "   Seeds: 42, 123, 456, 789, 1234, 2024, 3141, 4242, 5555, 6789"
    echo "   Total experiments: 320 (4 methods × 8 datasets × 10 seeds)"
    echo "   Estimated time: ~70 hours"
    echo ""
    
    print_header "Dry Run Complete"
    echo "To actually run the setup, use one of:"
    echo "  ./setup_ubuntu.sh --quick       # Quick replication (5 min, no LLM)"
    echo "  ./setup_ubuntu.sh --setup-only  # Install dependencies only"
    echo "  ./setup_ubuntu.sh               # Full setup + experiments"
    echo ""
}

# Main execution
print_header "QualSynth Ubuntu Setup Script"
echo "Script directory: $SCRIPT_DIR"
echo "Project directory: $PROJECT_DIR"
echo ""

# Handle dry run first
if [[ "$DRY_RUN" == true ]]; then
    run_dry_run
    exit 0
fi

if [[ "$QUICK_MODE" == true ]]; then
    # Quick mode - just run with pre-computed results
    check_system
    install_system_dependencies
    install_python
    setup_virtual_environment
    run_quick_replication
    exit 0
fi

if [[ "$RUN_ONLY" == false ]]; then
    # Full setup
    check_system
    install_system_dependencies
    install_python
    setup_virtual_environment
    install_ollama
    download_model
fi

if [[ "$SETUP_ONLY" == false ]]; then
    # Run experiments
    run_experiments
fi

print_header "Setup Complete!"
echo ""
echo "To activate the environment manually:"
echo "  source $PROJECT_DIR/venv/bin/activate"
echo ""
echo "To run quick replication:"
echo "  cd $PROJECT_DIR/replication && python replication.py --quick"
echo ""
echo "To run full experiments:"
echo "  cd $PROJECT_DIR/replication && python replication.py --full"
echo ""

