document.getElementById('updateButton').addEventListener('click', function() {
    console.log('Update button clicked');
});
document.getElementById('updateButton').addEventListener('click', function() {
    var learningRate = document.getElementById('learningRate').value;
    var optimizer = document.getElementById('optimizer').value;
    var gradClip = document.getElementById('gradClip').value;
    var weightDecay = document.getElementById('weightDecay').value;
    var momentum = optimizer === 'SGD' || optimizer === 'RMSprop' ? document.getElementById('momentum').value : undefined;
    //var beta1 = optimizer === 'Adam' || optimizer === 'AdamW' ? document.getElementById('beta1').value : undefined;
    //var beta2 = optimizer === 'Adam' || optimizer === 'AdamW' ? document.getElementById('beta2').value : undefined;
    //var eps = document.getElementById('eps').value;
    //var rho = optimizer === 'RMSprop' ? document.getElementById('rho').value : undefined;

    console.log('Updating model with the following parameters:');
    console.log('Learning rate:', learningRate);
    fetch('http://127.0.0.1:5000/update_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            learning_rate: learningRate,
            optimizer: optimizer,
            grad_clip: gradClip,
            weight_decay: weightDecay,
            momentum: momentum,
            //beta1: beta1,
            //beta2: beta2,
            //eps: eps,
            //rho: rho,
        }),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('statusMessage').innerText = 'Model updated successfully!';
        document.getElementById('statusMessage').style.display = 'block';
    })
    .catch((error) => {
        console.error('Error:', error);
        document.getElementById('statusMessage').innerText = 'Error updating model.';
        document.getElementById('statusMessage').style.display = 'block';
    });
});

// Listener for optimizer change to dynamically show/hide options
document.getElementById('optimizer').addEventListener('change', function() {
    var optimizer = this.value;
    // Hide all optional fields initially
    document.querySelectorAll('.optional').forEach(function(element) {
        element.style.display = 'none';
    });
    
    // Show specific fields based on the selected optimizer
    if (optimizer === 'SGD' || optimizer === 'RMSprop') {
        document.getElementById('momentum').parentNode.style.display = 'block';
    }
    if (optimizer === 'Adam' || optimizer === 'AdamW') {
        document.getElementById('beta1').parentNode.style.display = 'block';
        document.getElementById('beta2').parentNode.style.display = 'block';
    }
    if (optimizer === 'RMSprop') {
        document.getElementById('rho').parentNode.style.display = 'block';
    }
});

// Initial call to hide/show fields based on the default selected optimizer
document.getElementById('optimizer').dispatchEvent(new Event('change'));


