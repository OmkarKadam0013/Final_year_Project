# quickstart.sh
#!/bin/bash

echo "Quick Start - Dysarthric Speech Conversion"
echo "=========================================="
echo ""
echo "Choose an option:"
echo "  1. Run full setup"
echo "  2. Generate test data only"
echo "  3. Run tests only"
echo "  4. Train model (1 epoch test)"
echo "  5. Start backend server"
echo "  6. Start frontend"
echo "  7. Start both (backend + frontend)"
echo "  8. Run with Docker"
echo ""
read -p "Enter option (1-8): " option

case $option in
    1)
        echo "Running full setup..."
        chmod +x setup.sh
        ./setup.sh
        ;;
    2)
        echo "Generating test data..."
        source venv/bin/activate
        python scripts/generate_test_data.py --num-files 10
        ;;
    3)
        echo "Running tests..."
        source venv/bin/activate
        python scripts/test_local.py
        ;;
    4)
        echo "Training model (1 epoch)..."
        source venv/bin/activate
        python scripts/train.py --epochs 1 --batch-size 2
        ;;
    5)
        echo "Starting backend server..."
        source venv/bin/activate
        python -m backend.app.main
        ;;
    6)
        echo "Starting frontend..."
        cd frontend
        npm start
        ;;
    7)
        echo "Starting backend and frontend..."
        source venv/bin/activate
        python -m backend.app.main &
        BACKEND_PID=$!
        cd frontend
        npm start &
        FRONTEND_PID=$!
        echo ""
        echo "Backend PID: $BACKEND_PID"
        echo "Frontend PID: $FRONTEND_PID"
        echo ""
        echo "Press Ctrl+C to stop both"
        wait
        ;;
    8)
        echo "Starting with Docker..."
        docker-compose up --build
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac
