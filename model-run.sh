 #!/bin/bash

# Run model.py in the background
python cnn-model.py &

# Wait for the background process to finish
wait

# Check if mnist_model.pt file is generated
if [ -f "mnist_model.pt" ]; then
    echo "Model generated successfully. Starting app.py..." >> app.log
    # Run app.py
    python serving.py
else
    echo "Error: mnist_model.pt not found. Exiting." >> app.log
fi

