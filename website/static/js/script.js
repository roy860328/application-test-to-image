
// Function to send the text and get the image
function sendText() {
  var text = document.getElementById("textInput").value;
// script.js
  $.ajax({
    url: '/image',
    type: 'POST',
    data: JSON.stringify({ 'text': text }),
    contentType: 'application/json',
    success: function(response) {
        var imageUrl = response.image;
        $('#outputImage').attr('src', imageUrl);
    },
    error: function(xhr, status, error) {
        console.log('Error:', error);
    }
  });
  // // Call the API and retrieve the image
  // // Replace 'API_URL' with the actual API endpoint
  // fetch('/image', {
  //   method: 'POST',
  //   headers: {
  //     'Content-Type': 'application/json'
  //   },
  //   body: JSON.stringify({ text: text })
  // })
  //   .then(response => response.blob())
  //   .then(blob => {
  //     const imageUrl = URL.createObjectURL(blob);
  //     document.getElementById('outputImage').src = imageUrl;
  //   // .then(response => response.json())
  //   // .then(data => {
  //     // document.getElementById('outputImage').src = dataUrl; // Set the source of the output image element
  //     // const imageSrc = data.image;
  //     // const imageUrl = URL.createObjectURL(imageSrc);
  //     // document.getElementById('outputImage').src = imageUrl; // Set the source of the output image element

  //     // const imageSrc = data.image; // Get the image URL from the API response
  //     // document.getElementById('outputImage').src = imageSrc; // Set the source of the output image element
  //   })
  //   .catch(error => console.error(error));
}


