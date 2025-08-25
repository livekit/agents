
var createVADModule = (() => {
  var _scriptDir = import.meta.url;
  
  return (
function(createVADModule) {
  createVADModule = createVADModule || {};


var a;a||(a=typeof createVADModule !== 'undefined' ? createVADModule : {});var k,l;a.ready=new Promise(function(b,c){k=b;l=c});var p=Object.assign({},a),r="object"==typeof window,u="function"==typeof importScripts,v="",w;
if(r||u)u?v=self.location.href:"undefined"!=typeof document&&document.currentScript&&(v=document.currentScript.src),_scriptDir&&(v=_scriptDir),0!==v.indexOf("blob:")?v=v.substr(0,v.replace(/[?#].*/,"").lastIndexOf("/")+1):v="",u&&(w=b=>{var c=new XMLHttpRequest;c.open("GET",b,!1);c.responseType="arraybuffer";c.send(null);return new Uint8Array(c.response)});var aa=a.print||console.log.bind(console),x=a.printErr||console.warn.bind(console);Object.assign(a,p);p=null;var y;a.wasmBinary&&(y=a.wasmBinary);
var noExitRuntime=a.noExitRuntime||!0;"object"!=typeof WebAssembly&&z("no native wasm support detected");var A,B=!1,C="undefined"!=typeof TextDecoder?new TextDecoder("utf8"):void 0,D,E,F;function J(){var b=A.buffer;D=b;a.HEAP8=new Int8Array(b);a.HEAP16=new Int16Array(b);a.HEAP32=new Int32Array(b);a.HEAPU8=E=new Uint8Array(b);a.HEAPU16=new Uint16Array(b);a.HEAPU32=F=new Uint32Array(b);a.HEAPF32=new Float32Array(b);a.HEAPF64=new Float64Array(b)}var K=[],L=[],M=[];
function ba(){var b=a.preRun.shift();K.unshift(b)}var N=0,O=null,P=null;function z(b){if(a.onAbort)a.onAbort(b);b="Aborted("+b+")";x(b);B=!0;b=new WebAssembly.RuntimeError(b+". Build with -sASSERTIONS for more info.");l(b);throw b;}function Q(){return R.startsWith("data:application/octet-stream;base64,")}var R;if(a.locateFile){if(R="ten_vad.wasm",!Q()){var S=R;R=a.locateFile?a.locateFile(S,v):v+S}}else R=(new URL("ten_vad.wasm",import.meta.url)).href;
function T(){var b=R;try{if(b==R&&y)return new Uint8Array(y);if(w)return w(b);throw"both async and sync fetching of the wasm failed";}catch(c){z(c)}}function ca(){return y||!r&&!u||"function"!=typeof fetch?Promise.resolve().then(function(){return T()}):fetch(R,{credentials:"same-origin"}).then(function(b){if(!b.ok)throw"failed to load wasm binary file at '"+R+"'";return b.arrayBuffer()}).catch(function(){return T()})}function U(b){for(;0<b.length;)b.shift()(a)}
var da=[null,[],[]],ea={a:function(){z("")},f:function(b,c,m){E.copyWithin(b,c,c+m)},c:function(b){var c=E.length;b>>>=0;if(2147483648<b)return!1;for(var m=1;4>=m;m*=2){var h=c*(1+.2/m);h=Math.min(h,b+100663296);var d=Math;h=Math.max(b,h);d=d.min.call(d,2147483648,h+(65536-h%65536)%65536);a:{try{A.grow(d-D.byteLength+65535>>>16);J();var e=1;break a}catch(W){}e=void 0}if(e)return!0}return!1},e:function(){return 52},b:function(){return 70},d:function(b,c,m,h){for(var d=0,e=0;e<m;e++){var W=F[c>>2],
X=F[c+4>>2];c+=8;for(var G=0;G<X;G++){var f=E[W+G],H=da[b];if(0===f||10===f){f=H;for(var n=0,q=n+NaN,t=n;f[t]&&!(t>=q);)++t;if(16<t-n&&f.buffer&&C)f=C.decode(f.subarray(n,t));else{for(q="";n<t;){var g=f[n++];if(g&128){var I=f[n++]&63;if(192==(g&224))q+=String.fromCharCode((g&31)<<6|I);else{var Y=f[n++]&63;g=224==(g&240)?(g&15)<<12|I<<6|Y:(g&7)<<18|I<<12|Y<<6|f[n++]&63;65536>g?q+=String.fromCharCode(g):(g-=65536,q+=String.fromCharCode(55296|g>>10,56320|g&1023))}}else q+=String.fromCharCode(g)}f=q}(1===
b?aa:x)(f);H.length=0}else H.push(f)}d+=X}F[h>>2]=d;return 0}};
(function(){function b(d){a.asm=d.exports;A=a.asm.g;J();L.unshift(a.asm.h);N--;a.monitorRunDependencies&&a.monitorRunDependencies(N);0==N&&(null!==O&&(clearInterval(O),O=null),P&&(d=P,P=null,d()))}function c(d){b(d.instance)}function m(d){return ca().then(function(e){return WebAssembly.instantiate(e,h)}).then(function(e){return e}).then(d,function(e){x("failed to asynchronously prepare wasm: "+e);z(e)})}var h={a:ea};N++;a.monitorRunDependencies&&a.monitorRunDependencies(N);if(a.instantiateWasm)try{return a.instantiateWasm(h,
b)}catch(d){x("Module.instantiateWasm callback failed with error: "+d),l(d)}(function(){return y||"function"!=typeof WebAssembly.instantiateStreaming||Q()||"function"!=typeof fetch?m(c):fetch(R,{credentials:"same-origin"}).then(function(d){return WebAssembly.instantiateStreaming(d,h).then(c,function(e){x("wasm streaming compile failed: "+e);x("falling back to ArrayBuffer instantiation");return m(c)})})})().catch(l);return{}})();
a.___wasm_call_ctors=function(){return(a.___wasm_call_ctors=a.asm.h).apply(null,arguments)};a._malloc=function(){return(a._malloc=a.asm.i).apply(null,arguments)};a._free=function(){return(a._free=a.asm.j).apply(null,arguments)};a._ten_vad_create=function(){return(a._ten_vad_create=a.asm.k).apply(null,arguments)};a._ten_vad_process=function(){return(a._ten_vad_process=a.asm.l).apply(null,arguments)};a._ten_vad_destroy=function(){return(a._ten_vad_destroy=a.asm.m).apply(null,arguments)};
a._ten_vad_get_version=function(){return(a._ten_vad_get_version=a.asm.n).apply(null,arguments)};var V;P=function fa(){V||Z();V||(P=fa)};
function Z(){function b(){if(!V&&(V=!0,a.calledRun=!0,!B)){U(L);k(a);if(a.onRuntimeInitialized)a.onRuntimeInitialized();if(a.postRun)for("function"==typeof a.postRun&&(a.postRun=[a.postRun]);a.postRun.length;){var c=a.postRun.shift();M.unshift(c)}U(M)}}if(!(0<N)){if(a.preRun)for("function"==typeof a.preRun&&(a.preRun=[a.preRun]);a.preRun.length;)ba();U(K);0<N||(a.setStatus?(a.setStatus("Running..."),setTimeout(function(){setTimeout(function(){a.setStatus("")},1);b()},1)):b())}}
if(a.preInit)for("function"==typeof a.preInit&&(a.preInit=[a.preInit]);0<a.preInit.length;)a.preInit.pop()();Z();


  return createVADModule.ready
}
);
})();
export default createVADModule;