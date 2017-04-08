var wrapper = document.getElementById("signature-pad"),
    clearButton = wrapper.querySelector("[data-action=clear]"),
    submitButton = wrapper.querySelector("[data-action=submit]"),
    canvas = wrapper.querySelector("canvas"),
    signaturePad;

// Adjust canvas coordinate space taking into account pixel ratio,
// to make it look crisp on mobile devices.
// This also causes canvas to be cleared.
function resizeCanvas() {
    // When zoomed out to less than 100%, for some very strange reason,
    // some browsers report devicePixelRatio as less than 1
    // and only part of the canvas is cleared then.
    var ratio =  Math.max(window.devicePixelRatio || 1, 1);
    canvas.width = canvas.offsetWidth * ratio;
    canvas.height = canvas.offsetHeight * ratio;
    canvas.getContext("2d").scale(ratio, ratio);
}
function dataURItoBlob(dataURI) {
    var binary = atob(dataURI.split(',')[1]);
    var array = [];
    for(var i = 0; i < binary.length; i++) {
        array.push(binary.charCodeAt(i));
    }
    return new Blob([new Uint8Array(array)], {type: 'image/png'});
}
window.onresize = resizeCanvas;
resizeCanvas();

signaturePad = new SignaturePad(canvas);

clearButton.addEventListener("click", function (event) {
    signaturePad.clear();
});

// $(document).ready(function(e){
//     $("#my_form").submit(function(){
//         e.preventDefault();
//
//         if (signaturePad.isEmpty()) {
//             alert("Please skecth before submitting");
//         } else {
//             var i = signaturePad.toDataURL();
//             // window.open(signaturePad.toDataURL());
//             $.ajax({
//                 type: "POST",
//                 url:"sketchpad/",
//                 data:{csrfmiddlewaretoken:'{{ csrf_token }}', ge:'hhhhhhh'},
//                 success:handleResult,
//                 error:handleError
//             });
//         }
//     });
// });
///sample
// $(document).ready(function(){
//
//   $("#my_form").submit(function(){
//     $.post("",
//     {name:"Donald Duck",
//      city:"Duckburg",
//      csrfmiddlewaretoken:'{{ csrf_token }}'
//      },
//     function(data,status){
//       alert("Data: " + data + "\nStatus: " + status);
//     })
//     .fail(function(xhr) {
//         console.log("Error: " + xhr.statusText);
//         alert("Error: " + xhr.statusText);
//     });
//     return false;
//   });
//
// });
submitButton.addEventListener("click", function (event) {
    if (signaturePad.isEmpty()) {
        alert("Please skecth before submitting" + document.URL);
    } else {
        var dataURL = signaturePad.toDataURL('image/jpeg', 0.5);
        var blob = dataURItoBlob(dataURL);
        var fd = new FormData();
        // fd.append("canvasFile", blob);
        canvas.toBlob( function(blob) {
  fd.set("image0", blob, "image0.jpg");
}, "image/jpeg", 0.7);
// var xhr = new XMLHttpRequest();
// xhr.open('POST', document.URL, true);
// xhr.setRequestHeader("X-CSRFToken",'{{ csrf_token }}');
// xhr.send(fd);
$.post(document.URL, {image:signaturePad.toDataURL()}, function(data){
       $('#results').html(data);

   });
        //         jQuery.ajax({
        //     url: document.URL,
        //     data: {iimage:signaturePad.toDataURL()},
        //     cache: false,
        //     contentType: false,
        //     processData: false,
        //     type: 'POST',
        //     success: function(data){
        //         alert(data);
        //         // $('#results').html(data);
        //     }
        // });
        // $.get(document.URL, fd, function(data){
        //        $('#results').html(data);
        //
        //    });
        // $.get(document.URL, {image:signaturePad.toDataURL()}, function(data){
        //        $('#results').html(data);
        //
        //    });
        // window.open(signaturePad.toDataURL());
        // $.ajax({
        //     type: "POST",
        //     url:"sketchpad/",
        //     dataType: 'json',
        //     contentType: 'application/json; charset=utf-8',
        //     data:{image: signaturePad.toDataURL(), csrfmiddlewaretoken:'{{ csrf_token }}'},
        //     success:handleResult,
        //     error:handleError
        // });
    }
});

function handleResult(result,status, msg)
{
    // console.log(result); // log the returned json to the console
    // console.log("success"); // another sanity check
    alert(result);
        // $("#resultfield").html("Status:"+status+"<br>"+"Result:"+$(result).find("result").text()+"<br>");
        // $("#results").empty();
        // var l1= $(result).find("l1");
        // var perc1= $(result).find("p1");
        // var l2= $(result).find("l2");
        // var perc2= $(result).find("p2");
        // var l3= $(result).find("l3");
        // var perc3= $(result).find("l3");
        // var l4= $(result).find("l4");
        // var perc4= $(result).find("p4");
        // $("#results").append('<h2>Object recognised as <span style="color:green;">'+l1+'</span></h2><p>Top four categories for the above skecth:</p>');
        // $("#results").append('<div class="progress"><div class="progress-bar progress-bar-success" role="progressbar" aria-valuenow="'+p1+'" aria-valuemin="0" aria-valuemax="100" style="width:40%">'+p1+'% '+l1+'</div></div>');
        // $("#results").append('<div class="progress"><div class="progress-bar progress-bar-info" role="progressbar" aria-valuenow="'+p2+'" aria-valuemin="0" aria-valuemax="100" style="width:40%">'+p2+'% '+l2+'</div></div>');
        // $("#results").append('<div class="progress"><div class="progress-bar progress-bar-warning" role="progressbar" aria-valuenow="'+p3+'" aria-valuemin="0" aria-valuemax="100" style="width:40%">'+p3+'% '+l3+'</div></div>');
        // $("#results").append('<div class="progress"><div class="progress-bar progress-bar-danger" role="progressbar" aria-valuenow="'+p4+'" aria-valuemin="0" aria-valuemax="100" style="width:40%">'+p4+'% '+l4+'</div></div>');
}
function handleError(xhr,status,errmsg)
{
    $("#results").text("error:"+status+","+errmsg + "," + xhr);
}
