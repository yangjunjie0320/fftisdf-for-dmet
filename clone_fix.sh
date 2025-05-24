#!/bin/bash

echo "=== Git Clone Troubleshooting Script ==="

# Function to check SSH connection
check_ssh() {
    echo "Checking SSH connection to GitHub..."
    if ssh -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "✅ SSH connection to GitHub is working"
        return 0
    else
        echo "❌ SSH connection failed"
        return 1
    fi
}

# Function to clone with SSH
clone_ssh() {
    echo "Attempting to clone with SSH..."
    git clone --recurse-submodules git@github.com:yangjunjie0320/fftisdf-for-dmet.git
}

# Function to clone with HTTPS
clone_https() {
    echo "Attempting to clone with HTTPS..."
    git clone --recurse-submodules https://github.com/yangjunjie0320/fftisdf-for-dmet.git
}

# Function to clone step by step
clone_stepwise() {
    echo "Attempting step-by-step clone..."
    
    # Clone main repository
    echo "1. Cloning main repository..."
    if git clone git@github.com:yangjunjie0320/fftisdf-for-dmet.git; then
        cd fftisdf-for-dmet
    else
        echo "Trying HTTPS for main repository..."
        git clone https://github.com/yangjunjie0320/fftisdf-for-dmet.git
        cd fftisdf-for-dmet
    fi
    
    # Initialize submodules
    echo "2. Initializing submodules..."
    git submodule init
    
    # Update each submodule individually
    echo "3. Updating submodules individually..."
    
    echo "   - Updating fftisdf-main..."
    if ! git submodule update src/fftisdf-main; then
        echo "   ⚠️  Failed to update fftisdf-main"
    fi
    
    echo "   - Updating libdmet2-main..."
    if ! git submodule update src/libdmet2-main; then
        echo "   ⚠️  Failed to update libdmet2-main (may require special access)"
    fi
    
    echo "   - Updating pyscf-forge-lnocc..."
    if ! git submodule update src/pyscf-forge-lnocc; then
        echo "   ⚠️  Failed to update pyscf-forge-lnocc"
    fi
    
    echo "4. Checking submodule status..."
    git submodule status
}

# Main execution
echo "Choose your preferred method:"
echo "1. Check SSH and try SSH clone"
echo "2. Try HTTPS clone directly"
echo "3. Step-by-step clone (recommended)"
echo "4. All methods"

read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        if check_ssh; then
            clone_ssh
        else
            echo "SSH failed. Please set up SSH keys or try HTTPS."
        fi
        ;;
    2)
        clone_https
        ;;
    3)
        clone_stepwise
        ;;
    4)
        echo "Trying all methods..."
        if check_ssh; then
            clone_ssh
        else
            echo "SSH failed, trying HTTPS..."
            clone_https
        fi
        if [ $? -ne 0 ]; then
            echo "Direct clone failed, trying step-by-step..."
            clone_stepwise
        fi
        ;;
    *)
        echo "Invalid choice. Using step-by-step method..."
        clone_stepwise
        ;;
esac

echo "=== Clone process completed ===" 