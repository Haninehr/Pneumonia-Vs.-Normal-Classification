// Design By
// - https://dribbble.com/shots/13992184-File-Uploader-Drag-Drop


//first of all we need to get the paramters from browser's local storage !
$(function(){
  let img_src=localStorage.getItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-.img-model-choosen-img-src');
  let model_id=localStorage.getItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-span-attr-id');
  let model_c_html=localStorage.getItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-.model-choose-html');
  let help_value=localStorage.getItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.help');
  if(img_src!=null){
      $(".action").find(".img-model-choosen img").attr("src",img_src);}
  
  if(model_id!=null){
  $(".action").find("span.img-model-choosen").attr("id",model_id);}

  if(model_c_html!=null){
  $(".action").find(".model-choose").html(model_c_html);}

  if(help_value!=null){
    $(".help").attr("data-model",help_value);
  }
  
});

$(function(){
  $(".action").on("click",".model-choose li",function(){
      
      if($(this).attr("id")=='DL'){
          $(".action").find(".img-model-choosen img").attr("src","DL.png");
          $(".action").find(".model-choose").html('<li id="ML"><img src="ml.png" alt="" width="32px">Machine Learning Model</li>');
          $(".action").find("span").attr("id","DL");
          $(".help").attr("data-model","DL");
          //use local storage to save parametrs !
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-.img-model-choosen-img-src', 'dl.png');
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-span-attr-id', 'DL');
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-.model-choose-html', '<li id="ML"><img src="ml.png" alt="" width="32px">Machine Learning Model</li>');
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.help','DL');
        }else if($(this).attr("id")=='ML'){
          $(".action").find(".img-model-choosen img").attr("src","ml.png");
          $(".action").find(".model-choose").html('<li id="DL"><img src="dl.png" alt="" width="32px">Deep Learning Model</li>');
          $(".action").find("span").attr("id","ML");
          $(".help").attr("data-model","ML");
          //use local storage to save parametrs !
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-.img-model-choosen-img-src', 'ml.png');
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-span-attr-id', 'ML');
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.action-.model-choose-html', '<li id="DL"><img src="dl.png" alt="" width="32px">Deep Learning Model</li>');
          localStorage.setItem('Paremetrs-Ai-NORMAL-vs-PENUMONIA-Floating-Btn-.help','ML');
        }
  });
});

$(function(){
  $('#dropZoon').on('drop', function(e) {
  e.preventDefault();
  $("#result").html('');
  const dataTransfer = e.originalEvent.dataTransfer;

  // Check for files in the drop event
  if (dataTransfer.items) {
      for (let i = 0; i < dataTransfer.items.length; i++) {
          if (dataTransfer.items[i].kind === 'file') {
              const file = dataTransfer.items[i].getAsFile();
              
              handleFile(file); // Process the file
              break; // Stop processing after the first file
          }
      }
  } else {
      // Fallback for older browsers
      const files = dataTransfer.files;
      if (files.length > 0) {
          alert('files.lenght : '+files.length)
          handleFile(files[0]); // Process the first file
      }
  }
});

function handleFile(file) {
  if (!file) {
      alert("No image loaded");
      return;
  }

  // Optionally, update the file input (but not typically necessary)
  const fileList = new DataTransfer();
  fileList.items.add(file);
  $('#fileInput')[0].files = fileList.files; // This will not work directly

}

});


// Select Upload-Area
const uploadArea = document.querySelector('#uploadArea')

// Select Drop-Zoon Area
const dropZoon = document.querySelector('#dropZoon');

// Loading Text
const loadingText = document.querySelector('#loadingText');

// Slect File Input 
const fileInput = document.querySelector('#fileInput');

// Select Preview Image
const previewImage = document.querySelector('#previewImage');

// File-Details Area
const fileDetails = document.querySelector('#fileDetails');

// Uploaded File
const uploadedFile = document.querySelector('#uploadedFile');

// Uploaded File Info
const uploadedFileInfo = document.querySelector('#uploadedFileInfo');

// Uploaded File  Name
const uploadedFileName = document.querySelector('.uploaded-file__name');

// Uploaded File Icon
const uploadedFileIconText = document.querySelector('.uploaded-file__icon-text');

// Uploaded File Counter
const uploadedFileCounter = document.querySelector('.uploaded-file__counter');

//prediction btn 
const predictMaskbtn = document.querySelector('#predictbtn')
// ToolTip Data
const toolTipData = document.querySelector('.upload-area__tooltip-data');

// Images Types
const imagesTypes = [
  "jpeg",
  "png",
  "jpg",
];

// Append Images Types Array Inisde Tooltip Data
toolTipData.innerHTML = [...imagesTypes].join(', .');

// When (drop-zoon) has (dragover) Event 
dropZoon.addEventListener('dragover', function (event) {
  // Prevent Default Behavior 
  event.preventDefault();

  // Add Class (drop-zoon--over) On (drop-zoon)
  dropZoon.classList.add('drop-zoon--over');
});

// When (drop-zoon) has (dragleave) Event 
dropZoon.addEventListener('dragleave', function (event) {
  // Remove Class (drop-zoon--over) from (drop-zoon)
  dropZoon.classList.remove('drop-zoon--over');
});

// When (drop-zoon) has (drop) Event 
dropZoon.addEventListener('drop', function (event) {
  // Prevent Default Behavior 
  event.preventDefault();

  // Remove Class (drop-zoon--over) from (drop-zoon)
  dropZoon.classList.remove('drop-zoon--over');

  // Select The Dropped File
  const file = event.dataTransfer.files[0];

  // Call Function uploadFile(), And Send To Her The Dropped File :)
  uploadFile(file);
});

// When (drop-zoon) has (click) Event 
dropZoon.addEventListener('click', function (event) {
  // Click The (fileInput)
  fileInput.click();
});

// When (fileInput) has (change) Event 
fileInput.addEventListener('change', function (event) {
  // Select The Chosen File
  const file = event.target.files[0];
  $("#result").html('');
  // Call Function uploadFile(), And Send To Her The Chosen File :)
  uploadFile(file);
});

// Upload File Function
function uploadFile(file) {
  // FileReader()
  const fileReader = new FileReader();
  // File Type 
  const fileType = file.type;
  // File Size 
  const fileSize = file.size;

  // If File Is Passed from the (File Validation) Function
  if (fileValidate(fileType, fileSize)) {
    // Add Class (drop-zoon--Uploaded) on (drop-zoon)
    dropZoon.classList.add('drop-zoon--Uploaded');

    // Show Loading-text
    loadingText.style.display = "block";
    // Hide Preview Image
    previewImage.style.display = 'none';

    // Remove Class (uploaded-file--open) From (uploadedFile)
    uploadedFile.classList.remove('uploaded-file--open');
    // Remove Class (uploaded-file__info--active) from (uploadedFileInfo)
    uploadedFileInfo.classList.remove('uploaded-file__info--active');

    // After File Reader Loaded 
    fileReader.addEventListener('load', function () {
      // After Half Second 
      setTimeout(function () {
        // Add Class (upload-area--open) On (uploadArea)
        uploadArea.classList.add('upload-area--open');

        // Hide Loading-text (please-wait) Element
        loadingText.style.display = "none";
        // Show Preview Image
        previewImage.style.display = 'block';

        //show prediction btn
        predictMaskbtn.style.display="inline-block"

        // Add Class (file-details--open) On (fileDetails)
        fileDetails.classList.add('file-details--open');
        // Add Class (uploaded-file--open) On (uploadedFile)
        uploadedFile.classList.add('uploaded-file--open');
        // Add Class (uploaded-file__info--active) On (uploadedFileInfo)
        uploadedFileInfo.classList.add('uploaded-file__info--active');
      }, 500); // 0.5s

      // Add The (fileReader) Result Inside (previewImage) Source
      previewImage.setAttribute('src', fileReader.result);

      // Add File Name Inside Uploaded File Name
      uploadedFileName.innerHTML = file.name;

      // Call Function progressMove();
      progressMove();
    });

    // Read (file) As Data Url 
    fileReader.readAsDataURL(file);
  } else { // Else

    this; // (this) Represent The fileValidate(fileType, fileSize) Function

  };
};

// Progress Counter Increase Function
function progressMove() {
  // Counter Start
  let counter = 0;

  // After 600ms 
  setTimeout(() => {
    // Every 100ms
    let counterIncrease = setInterval(() => {
      // If (counter) is equle 100 
      if (counter === 100) {
        // Stop (Counter Increase)
        clearInterval(counterIncrease);
      } else { // Else
        // plus 10 on counter
        counter = counter + 10;
        // add (counter) vlaue inisde (uploadedFileCounter)
        uploadedFileCounter.innerHTML = `${counter}%`
      }
    }, 100);
  }, 600);
};


// Simple File Validate Function
function fileValidate(fileType, fileSize) {
  // File Type Validation
  let isImage = imagesTypes.filter((type) => fileType.indexOf(`image/${type}`) !== -1);

  // If The Uploaded File Type Is 'jpeg'
  if (isImage[0] === 'jpeg') {
    // Add Inisde (uploadedFileIconText) The (jpg) Value
    uploadedFileIconText.innerHTML = 'jpg';
  } else { // else
    // Add Inisde (uploadedFileIconText) The Uploaded File Type 
    uploadedFileIconText.innerHTML = isImage[0];
  };

  // If The Uploaded File Is An Image
  if (isImage.length !== 0) {
    // Check, If File Size Is 5MB or Less
    if (fileSize <= 5000000) { // 5MB :)
      return true;
    } else { // Else File Size
      return alert('Please Your File Should be 5 Megabytes or Less');
    };
  } else { // Else File Type 
    return alert('Please make sure to upload An Image File Type');
  };
};

// :)

function actionToggle() {
  const action = document.querySelector('.action');
  action.classList.toggle('active')
}

$(function(){
  $('#predictbtn').on('click', function() {
      $("#result").html('');
      $('#predictbtn').hide();
      $("#result").html('<div class="loading"><div class="loading-bar loading-1"></div><div class="loading-bar loading-2"></div>'+
'<div class="loading-bar loading-3"></div><div class="loading-bar loading-4"></div><div class="loading-bar loading-5"></div></div>')
      let formData = new FormData();
      
      let imageFile = $('#fileInput')[0].files[0];  // Get uploaded image
     
      formData.append('image', imageFile);

      var send = $(".action").find('span').attr("id");
      if(send=='ML' || send=='DL'){
          formData.append('model', send);  // Append the additional variable

          $.ajax({
          url: 'http://127.0.0.1:8000/predict-pneumonia',  // Flask URL
          type: 'POST',
          data: formData,
          processData: false,
          contentType: false,
          success: function(response) {
              //$('#result').text('Prediction: ' + response.result+' with Proba: '+response.Proba);
              //$("#result .Mask_Proba").show();
              RS=response.result;
              RB=response.confidence;
              //alert(RS);
              //alert(RB);
              rlt_to_show='<span class="Final-decision">'+'<span class="Final-decision-val fw-700';
              if(RS=='NORMAL'){
                  RS="Non Malade";
                  RB=(RB)*100;
                  RB=Math.round((RB -5 ) * 100) / 100
                  rlt_to_show+=' Green';
                  
                  
              }else{
                RS="Malade";
                RB=100-((RB)*100);
                RB=Math.round((RB -5) * 100) / 100;
                  rlt_to_show+=' Red';
              }
              rlt_to_show+='">'+RS+'</span> avec une confiance de <span class="fw-700">'+RB+'%</span></span>';
              setTimeout(() => {
                  $("#result").html(rlt_to_show);
              },3500);
             
          },
          error: function(err) {
              console.error('Error in prediction:', err);
          }
      });
      }else{
          //no model chosen ( no ml or dl !)
      }
      
  });
});