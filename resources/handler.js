const multiscaleVideo = document.getElementById("multiscale-video");
const ficus = document.getElementById("ficus");
const lego = document.getElementById("lego");
const mic = document.getElementById("mic");
const ship = document.getElementById("ship");

ficus.addEventListener("click", function(e){
  multiscaleVideo.src = "./vids/ficus.mp4";
  multiscaleVideo.poster = "./images/ficus.jpg";
  ficus.className = "selected-scene";
  lego.className = "";
  mic.className = "";
  ship.className = "";
});

lego.addEventListener("click", function(e){
  multiscaleVideo.src = "./vids/lego.mp4";
  multiscaleVideo.poster = "./images/lego.jpg";
  ficus.className = "";
  lego.className = "selected-scene";
  mic.className = "";
  ship.className = "";
});

mic.addEventListener("click", function(e){
  multiscaleVideo.src = "./vids/mic.mp4";
  multiscaleVideo.poster = "./images/mic.jpg";
  ficus.className = "";
  lego.className = "";
  mic.className = "selected-scene";
  ship.className = "";
});

ship.addEventListener("click", function(e){
  multiscaleVideo.src = "./vids/ship.mp4";
  multiscaleVideo.poster = "./images/ship.jpg";
  ficus.className = "";
  lego.className = "";
  mic.className = "";
  ship.className = "selected-scene";
});

multiscaleVideo.poster = "./images/ficus.jpg";

const blenderAVideo = document.getElementById("blender-a-video");
const checkerboard = document.getElementById("checkerboard");
const brick = document.getElementById("brick");

checkerboard.addEventListener("click", function(e){
  blenderAVideo.src = "./vids/checkerboard.mp4";
  blenderAVideo.poster = "./images/checkerboard.jpg";
  checkerboard.className = "selected-scene";
  brick.className = "";
});

brick.addEventListener("click", function(e){
  blenderAVideo.src = "./vids/brick.mp4";
  blenderAVideo.poster =  " ./images/brick.jpg";
  checkerboard.className = "";
  brick.className = "selected-scene";
});

blenderAVideo.poster = "./images/checkerboard.jpg";

const boatVideo = document.getElementById("boat-video");
const boatNearest = document.getElementById("boat-nearest");
const boatComp = document.getElementById("boat-comp");

boatNearest.addEventListener("click", function(e){
  boatVideo.src = "./vids/boat-nearest.mp4";
  boatVideo.poster = "./images/boat-nearest.jpg";
  boatNearest.className = "selected-scene";
  boatComp.className = "";
});

boatComp.addEventListener("click", function(e){
  boatVideo.src = "./vids/boat-comp.mp4";
  boatVideo.poster = "./images/boat-comp.jpg";
  boatNearest.className = "";
  boatComp.className = "selected-scene";
});

boatVideo.poster = "./images/boat-nearest.jpg";