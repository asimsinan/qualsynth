#!/bin/bash

# Run experiments with Ollama on M4 chip (GPU-accelerated)
# Automatically detects LLM methods and sets up Ollama only when needed
# Usage: ./scripts/run_with_ollama_m4.sh <experiment_name> [args...]

set -e

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║        🚀 UNIFIED EXPERIMENT RUNNER (LLM + Non-LLM Methods) 🚀          ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Extract methods from arguments
METHODS_LIST=()
COLLECT_METHODS=false
for arg in "$@"; do
    if [ "$COLLECT_METHODS" = true ]; then
        if [[ "$arg" == --* ]]; then
            COLLECT_METHODS=false
        else
            METHODS_LIST+=("$arg")
            continue
        fi
    fi
    if [[ "$arg" == "--methods" ]]; then
        COLLECT_METHODS=true
    fi
done

# Check if any LLM methods are present
LLM_METHODS=("qualsynth")
HAS_LLM_METHOD=false

if [ ${#METHODS_LIST[@]} -gt 0 ]; then
    for method in "${METHODS_LIST[@]}"; do
        for llm_method in "${LLM_METHODS[@]}"; do
            if [[ "$method" == "$llm_method"* ]]; then
                HAS_LLM_METHOD=true
                break 2
            fi
        done
    done
fi

# Step 1: Check system
echo "Step 1/6: Checking system..."
CHIP=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
echo "   Chip: $CHIP"

if [[ "$CHIP" == *"M4"* ]]; then
    echo "   ✅ M4 chip detected! GPU acceleration will be used for LLM methods."
elif [[ "$CHIP" == *"M1"* ]] || [[ "$CHIP" == *"M2"* ]] || [[ "$CHIP" == *"M3"* ]]; then
    echo "   ⚠️  Detected $CHIP (not M4). Script will still work but may not be fully optimized."
    echo "   Consider using ./scripts/run_with_ollama.sh instead."
else
    echo "   ⚠️  Could not detect Apple Silicon. This script is optimized for M4."
    echo "   Consider using ./scripts/run_with_ollama.sh instead."
fi
echo ""

# Step 2: Check if LLM methods are present
echo "Step 2/6: Detecting method types..."
if [ ${#METHODS_LIST[@]} -gt 0 ]; then
    echo "   Methods specified: ${METHODS_LIST[*]}"
else
    echo "   Methods: All from config (will detect LLM methods during execution)"
fi

if [ "$HAS_LLM_METHOD" = true ]; then
    echo "   ✅ LLM methods detected (qualsynth)"
    echo "   → Will set up Ollama for LLM methods"
    SETUP_OLLAMA=true
else
    echo "   ℹ️  No LLM methods detected"
    echo "   → Will skip Ollama setup (non-LLM methods only)"
    SETUP_OLLAMA=false
fi
echo ""

# Step 3: Set up Ollama only if LLM methods are present
if [ "$SETUP_OLLAMA" = true ]; then
    echo "Step 3/6: Setting up Ollama for LLM methods..."
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "   ✅ Ollama server is running"
    else
        echo "   ❌ Ollama server is not running!"
        echo ""
        echo "   Starting Ollama server..."
        nohup ollama serve > /tmp/ollama_serve.log 2>&1 &
        sleep 3
        if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "   ✅ Ollama server started successfully"
        else
            echo "   ❌ Failed to start Ollama server"
            echo ""
            echo "   Please start it manually:"
            echo "     ollama serve"
            echo ""
            exit 1
        fi
    fi
    
    # Use the optimized M4 model (faster version with reduced context)
    MODEL_NAME="gemma3-m4-faster"
    echo "   Using M4-optimized model: $MODEL_NAME"
    
    echo "   Verifying model exists..."
    if ! ollama list | grep -q "gemma3-m4-faster"; then
        echo "   ❌ ERROR: Model 'gemma3-m4-faster' not found"
        echo ""
        echo "   The optimized model is required for LLM methods."
        echo "   Please create it first:"
        echo "     ollama create gemma3-m4-faster -f configs/ollama/gemma3-m4-faster.Modelfile"
        echo ""
        exit 1
    fi
    echo "   ✅ Model found: $MODEL_NAME"
    
    # Set environment variables
    export OPENAI_API_BASE="http://localhost:11434/v1"
    export OPENAI_API_KEY="${OPENAI_API_KEY:-ollama-local}"
    export OLLAMA_MODEL="$MODEL_NAME"
    
    echo "   ✅ Environment variables set:"
    echo "      OPENAI_API_BASE: $OPENAI_API_BASE"
    echo "      OPENAI_API_KEY: $OPENAI_API_KEY"
    echo "      OLLAMA_MODEL: $OLLAMA_MODEL"
else
    echo "Step 3/6: Skipping Ollama setup (no LLM methods)"
    echo "   ℹ️  Non-LLM methods (CTGAN, SMOTE, TabFairGDT) don't need Ollama"
fi
echo ""

# Step 4: Verify environment variable (if Ollama was set up)
if [ "$SETUP_OLLAMA" = true ]; then
    echo "Step 4/6: Verifying OLLAMA_MODEL environment variable..."
    if [ "$OLLAMA_MODEL" != "$MODEL_NAME" ]; then
        echo "   ❌ ERROR: OLLAMA_MODEL not set correctly"
        echo "      Expected: $MODEL_NAME"
        echo "      Got: $OLLAMA_MODEL"
        exit 1
    fi
    echo "   ✅ OLLAMA_MODEL verified: $OLLAMA_MODEL"
else
    echo "Step 4/6: Skipping Ollama verification (no LLM methods)"
fi
echo ""

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                           ║"
echo "║                    ✅ ALL CHECKS PASSED ✅                                 ║"
echo "║                                                                           ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🔧 Configuration Summary:"
echo "  ┌─────────────────────────────────────────────────────────────────────┐"
if [ "$SETUP_OLLAMA" = true ]; then
    echo "  │ LLM Methods:     Enabled (qualsynth)                                  │"
    echo "  │ Server:          http://localhost:11434/v1                        │"
    echo "  │ Model:           $MODEL_NAME                                       │"
    echo "  │ Chip:            $CHIP                                             │"
    echo "  │                                                                      │"
    echo "  │ LLM Features:                                                       │"
    echo "  │   ✓ 131K context window (num_ctx=131072)                            │"
    echo "  │   ✓ 32K output tokens (num_predict=32768)                           │"
    echo "  │   ✓ GPU acceleration (num_gpu=999, all layers on M4)                │"
    echo "  │   ✓ Minimal CPU usage (num_thread=4)                                │"
    if [[ "$MODEL_NAME" == *"fast"* ]]; then
        echo "  │   ✓ Performance optimized (batch=512, +20% faster)                  │"
    fi
    echo "  │   ✓ High diversity (temperature=0.9, repeat_penalty=1.1)            │"
    echo "  │   ✓ Memory optimized (mmap, mlock, FP16 KV cache)                   │"
else
    echo "  │ LLM Methods:     Not detected (non-LLM methods only)                │"
    echo "  │ Chip:            $CHIP                                             │"
fi
echo "  └─────────────────────────────────────────────────────────────────────┘"
echo ""
if [ "$SETUP_OLLAMA" = true ]; then
    echo "📊 Expected Performance (M4 GPU vs CPU-only):"
    echo "  ┌─────────────────────────────────────────────────────────────────────┐"
    echo "  │                        CPU Only    M4 GPU    Speedup                │"
    echo "  ├─────────────────────────────────────────────────────────────────────┤"
    echo "  │ Tokens/second:         ~10         ~50       5x                     │"
    echo "  │ Per iteration:         3.5 min     42 sec    5x                     │"
    echo "  │ 12 iterations:         42 min      8-9 min   4.7x                   │"
    echo "  │ Ablation (9 exp):      6.3 hrs     1.4 hrs   4.5x                   │"
    echo "  └─────────────────────────────────────────────────────────────────────┘"
    echo ""
    echo "⚠️  NOTE: LLM methods will use M4-optimized model"
fi
echo ""

# Set up log file
LOG_DIR="results/experiment_runs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/ollama_m4_run_${TIMESTAMP}.log"

echo "📝 Log file: $LOG_FILE"
echo ""

# Step 5: Run iteration predictor (only for LLM methods)
if [ "$SETUP_OLLAMA" = true ]; then
    echo "Step 5/6: Running Adaptive Iteration Predictor for LLM methods..."
    
    # Check if --max-iterations is already provided
    MAX_ITERATIONS_PROVIDED=false
    for arg in "$@"; do
        if [[ "$arg" == "--max-iterations" ]]; then
            MAX_ITERATIONS_PROVIDED=true
            break
        fi
    done
    
    # If max-iterations not provided, run predictor to get optimal value
    if [ "$MAX_ITERATIONS_PROVIDED" = false ]; then
        echo "   🔮 Predicting optimal iterations..."
        
        # Extract dataset from arguments (look for --datasets argument)
        DATASET=""
        NEXT_IS_DATASET=false
        for arg in "$@"; do
            if [ "$NEXT_IS_DATASET" = true ]; then
                DATASET="$arg"
                break
            fi
            if [[ "$arg" == "--datasets" ]]; then
                NEXT_IS_DATASET=true
            fi
        done
        
        # Default to german_credit if not specified
        if [ -z "$DATASET" ]; then
            DATASET="german_credit"
        fi
        
        # Run predictor and capture output (with error handling)
        cd "$(dirname "$0")/.."
        
        # Check if predictor file exists first
        if [ ! -f "src/qualsynth/utils/adaptive_iteration_predictor.py" ]; then
            echo "   ⚠️  Adaptive iteration predictor not found, using default: 15 iterations"
            PREDICTED_ITERATIONS=15
        else
            PREDICTOR_OUTPUT=$(python3 -c "
import sys
from pathlib import Path
project_root = Path('$PWD')
sys.path.insert(0, str(project_root))

from src.qualsynth.data.splitting import load_split
from src.qualsynth.utils.adaptive_iteration_predictor import AdaptiveIterationPredictor

# Load data
split_data = load_split('$DATASET', seed=42, return_raw=True)
X_train = split_data['X_train']
y_train = split_data['y_train']
sensitive_features = None  # Not needed for iteration prediction

# Run SOTA predictor (ALWAYS targets 1:1 balance)
predictor = AdaptiveIterationPredictor(
    target_samples=None,  # Auto-calculate 1:1 balance
    max_samples=10000,    # High cap to not limit 1:1 balance
    max_time_hours=2.0,   # 2-hour time limit
    batch_size=20,        # Match config batch_size
    duplicate_threshold=0.10,
    quality_threshold=0.3,
    verbose=False         # Suppress verbose output in script
)

result = predictor.predict(X_train, y_train, sensitive_features, '$DATASET')
# Print only the number (last line) - redirect verbose output to stderr
print(result.predicted_iterations)
" 2>&1 | tail -1)
            
            if [ -n "$PREDICTOR_OUTPUT" ] && [ "$PREDICTOR_OUTPUT" -gt 0 ] 2>/dev/null; then
                PREDICTED_ITERATIONS=$PREDICTOR_OUTPUT
                echo "   ✅ Optimal iterations predicted: $PREDICTED_ITERATIONS"
            else
                echo "   ⚠️  Predictor failed or returned invalid value: '$PREDICTOR_OUTPUT'"
                echo "   Using default: 15 iterations"
                PREDICTED_ITERATIONS=15
            fi
        fi
        
        # Add --max-iterations to arguments
        set -- "$@" "--max-iterations" "$PREDICTED_ITERATIONS"
        echo "   🎯 Using iterations: $PREDICTED_ITERATIONS"
    else
        echo "   ℹ️  Using user-provided --max-iterations"
    fi
else
    echo "Step 5/6: Skipping iteration predictor (no LLM methods)"
fi
echo ""

# Step 6: Run experiments
echo "Step 6/6: Running experiments..."

# Run the experiment script with all arguments
echo "   🚀 Starting experiment execution..."
echo ""
python scripts/run_experiments.py "$@" 2>&1 | tee "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]:-0}

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    ✅ EXPERIMENTS COMPLETED ✅                             ║"
else
    echo "║                    ❌ EXPERIMENTS FAILED ❌                               ║"
fi
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📝 Full log saved to: $LOG_FILE"

exit $EXIT_CODE

