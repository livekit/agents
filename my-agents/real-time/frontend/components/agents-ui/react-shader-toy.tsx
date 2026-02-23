import React, { type ComponentPropsWithoutRef, useEffect, useRef } from 'react';

const PRECISIONS = ['lowp', 'mediump', 'highp'];
const FS_MAIN_SHADER = `\nvoid main(void){
    vec4 color = vec4(0.0,0.0,0.0,1.0);
    mainImage( color, gl_FragCoord.xy );
    gl_FragColor = color;
}`;
const BASIC_FS = `void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 uv = fragCoord/iResolution.xy;
    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));
    fragColor = vec4(col,1.0);
}`;
const BASIC_VS = `attribute vec3 aVertexPosition;
void main(void) {
    gl_Position = vec4(aVertexPosition, 1.0);
}`;
const UNIFORM_TIME = 'iTime';
const UNIFORM_TIMEDELTA = 'iTimeDelta';
const UNIFORM_DATE = 'iDate';
const UNIFORM_FRAME = 'iFrame';
const UNIFORM_MOUSE = 'iMouse';
const UNIFORM_RESOLUTION = 'iResolution';
const UNIFORM_CHANNEL = 'iChannel';
const UNIFORM_CHANNELRESOLUTION = 'iChannelResolution';
const UNIFORM_DEVICEORIENTATION = 'iDeviceOrientation';

type Vector4<T = number> = [T, T, T, T];
type UniformType = keyof Uniforms;

function isMatrixType(t: string, v: number[] | number): v is number[] {
  return t.includes('Matrix') && Array.isArray(v);
}
function isVectorListType(t: string, v: number[] | number): v is number[] {
  return t.includes('v') && Array.isArray(v) && v.length > Number.parseInt(t.charAt(0));
}
function isVectorType(t: string, v: number[] | number): v is Vector4 {
  return !t.includes('v') && Array.isArray(v) && v.length > Number.parseInt(t.charAt(0));
}
const processUniform = <T extends UniformType>(
  gl: WebGLRenderingContext,
  location: WebGLUniformLocation,
  t: T,
  value: number | number[]
) => {
  if (isVectorType(t, value)) {
    switch (t) {
      case '2f':
        return gl.uniform2f(location, value[0], value[1]);
      case '3f':
        return gl.uniform3f(location, value[0], value[1], value[2]);
      case '4f':
        return gl.uniform4f(location, value[0], value[1], value[2], value[3]);
      case '2i':
        return gl.uniform2i(location, value[0], value[1]);
      case '3i':
        return gl.uniform3i(location, value[0], value[1], value[2]);
      case '4i':
        return gl.uniform4i(location, value[0], value[1], value[2], value[3]);
    }
  }
  if (typeof value === 'number') {
    switch (t) {
      case '1i':
        return gl.uniform1i(location, value);
      default:
        return gl.uniform1f(location, value);
    }
  }
  switch (t) {
    case '1iv':
      return gl.uniform1iv(location, value);
    case '2iv':
      return gl.uniform2iv(location, value);
    case '3iv':
      return gl.uniform3iv(location, value);
    case '4iv':
      return gl.uniform4iv(location, value);
    case '1fv':
      return gl.uniform1fv(location, value);
    case '2fv':
      return gl.uniform2fv(location, value);
    case '3fv':
      return gl.uniform3fv(location, value);
    case '4fv':
      return gl.uniform4fv(location, value);
    case 'Matrix2fv':
      return gl.uniformMatrix2fv(location, false, value);
    case 'Matrix3fv':
      return gl.uniformMatrix3fv(location, false, value);
    case 'Matrix4fv':
      return gl.uniformMatrix4fv(location, false, value);
  }
};

const uniformTypeToGLSLType = (t: string) => {
  switch (t) {
    case '1f':
      return 'float';
    case '2f':
      return 'vec2';
    case '3f':
      return 'vec3';
    case '4f':
      return 'vec4';
    case '1i':
      return 'int';
    case '2i':
      return 'ivec2';
    case '3i':
      return 'ivec3';
    case '4i':
      return 'ivec4';
    case '1iv':
      return 'int';
    case '2iv':
      return 'ivec2';
    case '3iv':
      return 'ivec3';
    case '4iv':
      return 'ivec4';
    case '1fv':
      return 'float';
    case '2fv':
      return 'vec2';
    case '3fv':
      return 'vec3';
    case '4fv':
      return 'vec4';
    case 'Matrix2fv':
      return 'mat2';
    case 'Matrix3fv':
      return 'mat3';
    case 'Matrix4fv':
      return 'mat4';
    default:
      console.error(
        log(`The uniform type "${t}" is not valid, please make sure your uniform type is valid`)
      );
  }
};

const LinearFilter = 9729;
const NearestFilter = 9728;
const LinearMipMapLinearFilter = 9987;
const ClampToEdgeWrapping = 33071;
const RepeatWrapping = 10497;

class Texture {
  gl: WebGLRenderingContext;
  url?: string;
  wrapS?: number;
  wrapT?: number;
  minFilter?: number;
  magFilter?: number;
  source?: HTMLImageElement | HTMLVideoElement;
  pow2canvas?: HTMLCanvasElement;
  isLoaded = false;
  isVideo = false;
  flipY = -1;
  width = 0;
  height = 0;
  _webglTexture: WebGLTexture | null = null;
  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }
  updateTexture = (texture: WebGLTexture, video: HTMLVideoElement, flipY: number) => {
    const { gl } = this;
    const level = 0;
    const internalFormat = gl.RGBA;
    const srcFormat = gl.RGBA;
    const srcType = gl.UNSIGNED_BYTE;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flipY);
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, srcFormat, srcType, video);
  };
  setupVideo = (url: string) => {
    const video = document.createElement('video');
    let playing = false;
    let timeupdate = false;
    video.autoplay = true;
    video.muted = true;
    video.loop = true;
    video.crossOrigin = 'anonymous';
    const checkReady = () => {
      if (playing && timeupdate) {
        this.isLoaded = true;
      }
    };
    video.addEventListener(
      'playing',
      () => {
        playing = true;
        this.width = video.videoWidth || 0;
        this.height = video.videoHeight || 0;
        checkReady();
      },
      true
    );
    video.addEventListener(
      'timeupdate',
      () => {
        timeupdate = true;
        checkReady();
      },
      true
    );
    video.src = url;
    return video;
  };
  makePowerOf2 = <T extends HTMLCanvasElement | HTMLImageElement | ImageBitmap>(image: T): T => {
    if (
      image instanceof HTMLImageElement ||
      image instanceof HTMLCanvasElement ||
      image instanceof ImageBitmap
    ) {
      if (this.pow2canvas === undefined) this.pow2canvas = document.createElement('canvas');
      this.pow2canvas.width = 2 ** Math.floor(Math.log(image.width) / Math.LN2);
      this.pow2canvas.height = 2 ** Math.floor(Math.log(image.height) / Math.LN2);
      const context = this.pow2canvas.getContext('2d');
      context?.drawImage(image, 0, 0, this.pow2canvas.width, this.pow2canvas.height);
      console.warn(
        log(
          `Image is not power of two ${image.width} x ${image.height}. Resized to ${this.pow2canvas.width} x ${this.pow2canvas.height};`
        )
      );
      return this.pow2canvas as T;
    }
    return image;
  };
  load = async (textureArgs: TextureParams) => {
    const { gl } = this;
    const { url, wrapS, wrapT, minFilter, magFilter, flipY = -1 }: TextureParams = textureArgs;
    if (!url) {
      return Promise.reject(
        new Error(log('Missing url, please make sure to pass the url of your texture { url: ... }'))
      );
    }
    const isImage = /(\.jpg|\.jpeg|\.png|\.gif|\.bmp)$/i.exec(url);
    const isVideo = /(\.mp4|\.3gp|\.webm|\.ogv)$/i.exec(url);
    if (isImage === null && isVideo === null) {
      return Promise.reject(
        new Error(log(`Please upload a video or an image with a valid format (url: ${url})`))
      );
    }
    Object.assign(this, { url, wrapS, wrapT, minFilter, magFilter, flipY });
    const level = 0;
    const internalFormat = gl.RGBA;
    const width = 1;
    const height = 1;
    const border = 0;
    const srcFormat = gl.RGBA;
    const srcType = gl.UNSIGNED_BYTE;
    const pixel = new Uint8Array([255, 255, 255, 0]);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      level,
      internalFormat,
      width,
      height,
      border,
      srcFormat,
      srcType,
      pixel
    );
    if (isVideo) {
      const video = this.setupVideo(url);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      this._webglTexture = texture;
      this.source = video;
      this.isVideo = true;
      return video.play().then(() => this);
    }
    async function loadImage() {
      return new Promise((resolve, reject) => {
        const image = new Image();
        image.crossOrigin = 'anonymous';
        image.onload = () => {
          resolve(image);
        };
        image.onerror = () => {
          reject(new Error(log(`failed loading url: ${url}`)));
        };
        image.src = url ?? '';
      });
    }
    let image = (await loadImage()) as HTMLImageElement;
    let isPowerOf2 =
      (image.width & (image.width - 1)) === 0 && (image.height & (image.height - 1)) === 0;
    if (
      (textureArgs.wrapS !== ClampToEdgeWrapping ||
        textureArgs.wrapT !== ClampToEdgeWrapping ||
        (textureArgs.minFilter !== NearestFilter && textureArgs.minFilter !== LinearFilter)) &&
      !isPowerOf2
    ) {
      image = this.makePowerOf2(image);
      isPowerOf2 = true;
    }
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, flipY);
    gl.texImage2D(gl.TEXTURE_2D, level, internalFormat, srcFormat, srcType, image);
    if (
      isPowerOf2 &&
      textureArgs.minFilter !== NearestFilter &&
      textureArgs.minFilter !== LinearFilter
    ) {
      gl.generateMipmap(gl.TEXTURE_2D);
    }
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, this.wrapS || RepeatWrapping);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, this.wrapT || RepeatWrapping);
    gl.texParameteri(
      gl.TEXTURE_2D,
      gl.TEXTURE_MIN_FILTER,
      this.minFilter || LinearMipMapLinearFilter
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, this.magFilter || LinearFilter);
    this._webglTexture = texture;
    this.source = image;
    this.isVideo = false;
    this.isLoaded = true;
    this.width = image.width || 0;
    this.height = image.height || 0;
    return this;
  };
}

const log = (text: string) => `react-shaders: ${text}`;

const latestPointerClientCoords = (e: MouseEvent | TouchEvent) => {
  if ('changedTouches' in e) {
    const t = e.changedTouches[0];
    return [t?.clientX ?? 0, t?.clientY ?? 0];
  }
  return [e.clientX, e.clientY];
};

const lerpVal = (v0: number, v1: number, t: number) => v0 * (1 - t) + v1 * t;
const insertStringAtIndex = (currentString: string, string: string, index: number) =>
  index > 0
    ? currentString.substring(0, index) +
      string +
      currentString.substring(index, currentString.length)
    : string + currentString;

type Uniform = { type: string; value: number[] | number };
type Uniforms = Record<string, Uniform>;
type TextureParams = {
  url: string;
  wrapS?: number;
  wrapT?: number;
  minFilter?: number;
  magFilter?: number;
  flipY?: number;
};

export interface ReactShaderToyProps {
  /** Fragment shader GLSL code. */
  fs: string;

  /** Vertex shader GLSL code. */
  vs?: string;

  /**
   * Textures to be passed to the shader. Textures need to be squared or will be
   * automatically resized.
   *
   * Options default to:
   *
   * ```js
   * {
   *   minFilter: LinearMipMapLinearFilter,
   *   magFilter: LinearFilter,
   *   wrapS: RepeatWrapping,
   *   wrapT: RepeatWrapping,
   * }
   * ```
   *
   * See [textures in the docs](https://rysana.com/docs/react-shaders#textures)
   * for details.
   */
  textures?: TextureParams[];

  /**
   * Custom uniforms to be passed to the shader.
   *
   * See [custom uniforms in the
   * docs](https://rysana.com/docs/react-shaders#custom-uniforms) for details.
   */
  uniforms?: Uniforms;

  /**
   * Color used when clearing the canvas.
   *
   * See [the WebGL
   * docs](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/clearColor)
   * for details.
   */
  clearColor?: Vector4;

  /**
   * GLSL precision qualifier. Defaults to `'highp'`. Balance between
   * performance and quality.
   */
  precision?: 'highp' | 'lowp' | 'mediump';

  /** Custom inline style for canvas. */
  style?: React.CSSProperties;

  /** Customize WebGL context attributes. See [the WebGL docs](https://developer.mozilla.org/en-US/docs/Web/API/WebGLRenderingContext/getContextAttributes) for details. */
  contextAttributes?: Record<string, unknown>;

  /** Lerp value for `iMouse` built-in uniform. Must be between 0 and 1. */
  lerp?: number;

  /** Device pixel ratio. */
  devicePixelRatio?: number;

  /**
   * Callback for when the textures are done loading. Useful if you want to do
   * something like e.g. hide the canvas until textures are done loading.
   */
  onDoneLoadingTextures?: () => void;

  /** Custom callback to handle errors. Defaults to `console.error`. */
  onError?: (error: string) => void;

  /** Custom callback to handle warnings. Defaults to `console.warn`. */
  onWarning?: (warning: string) => void;
}

export function ReactShaderToy({
  fs,
  vs = BASIC_VS,
  textures = [],
  uniforms: propUniforms,
  clearColor = [0, 0, 0, 1],
  precision = 'highp',
  style,
  contextAttributes = {},
  lerp = 1,
  devicePixelRatio = 1,
  onDoneLoadingTextures,
  onError = console.error,
  onWarning = console.warn,
  ...canvasProps
}: ReactShaderToyProps & ComponentPropsWithoutRef<'canvas'>) {
  // Refs for WebGL state
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const glRef = useRef<WebGLRenderingContext | null>(null);
  const squareVerticesBufferRef = useRef<WebGLBuffer | null>(null);
  const shaderProgramRef = useRef<WebGLProgram | null>(null);
  const vertexPositionAttributeRef = useRef<number | undefined>(undefined);
  const animFrameIdRef = useRef<number | undefined>(undefined);
  const mousedownRef = useRef(false);
  const canvasPositionRef = useRef<DOMRect | undefined>(undefined);
  const timerRef = useRef(0);
  const lastMouseArrRef = useRef<number[]>([0, 0]);
  const texturesArrRef = useRef<Texture[]>([]);
  const lastTimeRef = useRef(0);
  const resizeObserverRef = useRef<ResizeObserver | undefined>(undefined);
  const uniformsRef = useRef<
    Record<
      string,
      { type: string; isNeeded: boolean; value?: number[] | number; arraySize?: string }
    >
  >({
    [UNIFORM_TIME]: { type: 'float', isNeeded: false, value: 0 },
    [UNIFORM_TIMEDELTA]: { type: 'float', isNeeded: false, value: 0 },
    [UNIFORM_DATE]: { type: 'vec4', isNeeded: false, value: [0, 0, 0, 0] },
    [UNIFORM_MOUSE]: { type: 'vec4', isNeeded: false, value: [0, 0, 0, 0] },
    [UNIFORM_RESOLUTION]: { type: 'vec2', isNeeded: false, value: [0, 0] },
    [UNIFORM_FRAME]: { type: 'int', isNeeded: false, value: 0 },
    [UNIFORM_DEVICEORIENTATION]: { type: 'vec4', isNeeded: false, value: [0, 0, 0, 0] },
  });
  const propsUniformsRef = useRef<Uniforms | undefined>(propUniforms);

  const setupChannelRes = (texture: TextureParams | Texture, id: number) => {
    const width = 'width' in texture ? (texture.width ?? 0) : 0;
    const height = 'height' in texture ? (texture.height ?? 0) : 0;
    const channelResUniform = uniformsRef.current.iChannelResolution;
    if (!channelResUniform) return;
    const channelResValue = Array.isArray(channelResUniform.value)
      ? channelResUniform.value
      : (channelResUniform.value = []);
    channelResValue[id * 3] = width * devicePixelRatio;
    channelResValue[id * 3 + 1] = height * devicePixelRatio;
    channelResValue[id * 3 + 2] = 0;
  };

  const initWebGL = () => {
    if (!canvasRef.current) return;
    glRef.current = (canvasRef.current.getContext('webgl', contextAttributes) ||
      canvasRef.current.getContext(
        'experimental-webgl',
        contextAttributes
      )) as WebGLRenderingContext | null;
    glRef.current?.getExtension('OES_standard_derivatives');
    glRef.current?.getExtension('EXT_shader_texture_lod');
  };

  const initBuffers = () => {
    const gl = glRef.current;
    squareVerticesBufferRef.current = gl?.createBuffer() ?? null;
    gl?.bindBuffer(gl.ARRAY_BUFFER, squareVerticesBufferRef.current);
    const vertices = [1.0, 1.0, 0.0, -1.0, 1.0, 0.0, 1.0, -1.0, 0.0, -1.0, -1.0, 0.0];
    gl?.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.STATIC_DRAW);
  };

  const onDeviceOrientationChange = ({ alpha, beta, gamma }: DeviceOrientationEvent) => {
    uniformsRef.current.iDeviceOrientation!.value = [
      alpha ?? 0,
      beta ?? 0,
      gamma ?? 0,
      window.orientation ?? 0,
    ];
  };

  const mouseDown = (e: MouseEvent | TouchEvent) => {
    const [clientX = 0, clientY = 0] = latestPointerClientCoords(e);
    const mouseX = clientX - (canvasPositionRef.current?.left ?? 0) - window.pageXOffset;
    const mouseY =
      (canvasPositionRef.current?.height ?? 0) -
      clientY -
      (canvasPositionRef.current?.top ?? 0) -
      window.pageYOffset;
    mousedownRef.current = true;
    const mouseValue = Array.isArray(uniformsRef.current.iMouse?.value)
      ? uniformsRef.current.iMouse.value
      : undefined;
    if (mouseValue) {
      mouseValue[2] = mouseX;
      mouseValue[3] = mouseY;
    }
    lastMouseArrRef.current[0] = mouseX;
    lastMouseArrRef.current[1] = mouseY;
  };

  const mouseMove = (e: MouseEvent | TouchEvent) => {
    canvasPositionRef.current = canvasRef.current?.getBoundingClientRect();
    const [clientX = 0, clientY = 0] = latestPointerClientCoords(e);
    const mouseX = clientX - (canvasPositionRef.current?.left ?? 0);
    const mouseY =
      (canvasPositionRef.current?.height ?? 0) - clientY - (canvasPositionRef.current?.top ?? 0);
    if (lerp !== 1) {
      lastMouseArrRef.current[0] = mouseX;
      lastMouseArrRef.current[1] = mouseY;
    } else {
      const mouseValue = Array.isArray(uniformsRef.current.iMouse?.value)
        ? uniformsRef.current.iMouse.value
        : undefined;
      if (mouseValue) {
        mouseValue[0] = mouseX;
        mouseValue[1] = mouseY;
      }
    }
  };

  const mouseUp = () => {
    const mouseValue = Array.isArray(uniformsRef.current.iMouse?.value)
      ? uniformsRef.current.iMouse.value
      : undefined;
    if (mouseValue) {
      mouseValue[2] = 0;
      mouseValue[3] = 0;
    }
  };

  const onResize = () => {
    const gl = glRef.current;
    if (!gl) return;
    canvasPositionRef.current = canvasRef.current?.getBoundingClientRect();
    // Force pixel ratio to be one to avoid expensive calculus on retina display.
    const realToCSSPixels = devicePixelRatio;
    const displayWidth = Math.floor((canvasPositionRef.current?.width ?? 1) * realToCSSPixels);
    const displayHeight = Math.floor((canvasPositionRef.current?.height ?? 1) * realToCSSPixels);
    gl.canvas.width = displayWidth;
    gl.canvas.height = displayHeight;
    if (uniformsRef.current.iResolution?.isNeeded && shaderProgramRef.current) {
      const rUniform = gl.getUniformLocation(shaderProgramRef.current, UNIFORM_RESOLUTION);
      gl.uniform2fv(rUniform, [gl.canvas.width, gl.canvas.height]);
    }
  };

  const createShader = (type: number, shaderCodeAsText: string) => {
    const gl = glRef.current;
    if (!gl) return null;
    const shader = gl.createShader(type);
    if (!shader) return null;
    gl.shaderSource(shader, shaderCodeAsText);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      onWarning?.(log(`Error compiling the shader:\n${shaderCodeAsText}`));
      const compilationLog = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      onError?.(log(`Shader compiler log: ${compilationLog}`));
    }
    return shader;
  };

  const initShaders = (fragmentShader: string, vertexShader: string) => {
    const gl = glRef.current;
    if (!gl) return;
    const fragmentShaderObj = createShader(gl.FRAGMENT_SHADER, fragmentShader);
    const vertexShaderObj = createShader(gl.VERTEX_SHADER, vertexShader);
    shaderProgramRef.current = gl.createProgram();
    if (!shaderProgramRef.current || !vertexShaderObj || !fragmentShaderObj) return;
    gl.attachShader(shaderProgramRef.current, vertexShaderObj);
    gl.attachShader(shaderProgramRef.current, fragmentShaderObj);
    gl.linkProgram(shaderProgramRef.current);
    if (!gl.getProgramParameter(shaderProgramRef.current, gl.LINK_STATUS)) {
      onError?.(
        log(
          `Unable to initialize the shader program: ${gl.getProgramInfoLog(shaderProgramRef.current)}`
        )
      );
      return;
    }
    gl.useProgram(shaderProgramRef.current);
    vertexPositionAttributeRef.current = gl.getAttribLocation(
      shaderProgramRef.current,
      'aVertexPosition'
    );
    gl.enableVertexAttribArray(vertexPositionAttributeRef.current);
  };

  const processCustomUniforms = () => {
    if (propUniforms) {
      for (const name of Object.keys(propUniforms)) {
        const uniform = propUniforms[name];
        if (!uniform) continue;
        const { value, type } = uniform;
        const glslType = uniformTypeToGLSLType(type);
        if (!glslType) continue;
        const tempObject: { arraySize?: string } = {};
        if (isMatrixType(type, value)) {
          const arrayLength = type.length;
          const val = Number.parseInt(type.charAt(arrayLength - 3));
          const numberOfMatrices = Math.floor(value.length / (val * val));
          if (value.length > val * val) tempObject.arraySize = `[${numberOfMatrices}]`;
        } else if (isVectorListType(type, value)) {
          tempObject.arraySize = `[${Math.floor(value.length / Number.parseInt(type.charAt(0)))}]`;
        }
        uniformsRef.current[name] = { type: glslType, isNeeded: false, value, ...tempObject };
      }
    }
  };

  const processTextures = () => {
    const gl = glRef.current;
    if (!gl) return;
    if (textures && textures.length > 0) {
      uniformsRef.current[`${UNIFORM_CHANNELRESOLUTION}`] = {
        type: 'vec3',
        isNeeded: false,
        arraySize: `[${textures.length}]`,
        value: [],
      };
      const texturePromisesArr = textures.map((texture: TextureParams, id: number) => {
        // Dynamically add textures uniforms.
        uniformsRef.current[`${UNIFORM_CHANNEL}${id}`] = {
          type: 'sampler2D',
          isNeeded: false,
        };
        setupChannelRes(texture, id);
        texturesArrRef.current[id] = new Texture(gl);
        return texturesArrRef.current[id]?.load(texture).then((t: Texture) => {
          setupChannelRes(t, id);
        });
      });
      Promise.all(texturePromisesArr)
        .then(() => {
          if (onDoneLoadingTextures) onDoneLoadingTextures();
        })
        .catch((e) => {
          onError?.(e);
          if (onDoneLoadingTextures) onDoneLoadingTextures();
        });
    } else if (onDoneLoadingTextures) onDoneLoadingTextures();
  };

  const preProcessFragment = (fragment: string) => {
    const isValidPrecision = PRECISIONS.includes(precision ?? 'highp');
    const precisionString = `precision ${isValidPrecision ? precision : PRECISIONS[1]} float;\n`;
    if (!isValidPrecision) {
      onWarning?.(
        log(
          `wrong precision type ${precision}, please make sure to pass one of a valid precision lowp, mediump, highp, by default you shader precision will be set to highp.`
        )
      );
    }
    let fragmentShader = precisionString
      .concat(`#define DPR ${devicePixelRatio.toFixed(1)}\n`)
      .concat(fragment.replace(/texture\(/g, 'texture2D('));
    for (const uniform of Object.keys(uniformsRef.current)) {
      if (fragment.includes(uniform)) {
        const u = uniformsRef.current[uniform];
        if (!u) continue;
        fragmentShader = insertStringAtIndex(
          fragmentShader,
          `uniform ${u.type} ${uniform}${u.arraySize || ''}; \n`,
          fragmentShader.lastIndexOf(precisionString) + precisionString.length
        );
        u.isNeeded = true;
      }
    }
    const isShadertoy = fragment.includes('mainImage');
    if (isShadertoy) fragmentShader = fragmentShader.concat(FS_MAIN_SHADER);
    return fragmentShader;
  };

  const setUniforms = (timestamp: number) => {
    const gl = glRef.current;
    if (!gl || !shaderProgramRef.current) return;
    const delta = lastTimeRef.current ? (timestamp - lastTimeRef.current) / 1000 : 0;
    lastTimeRef.current = timestamp;
    const propUniforms = propsUniformsRef.current;
    if (propUniforms) {
      for (const name of Object.keys(propUniforms)) {
        const currentUniform = propUniforms[name];
        if (!currentUniform) continue;
        if (uniformsRef.current[name]?.isNeeded) {
          if (!shaderProgramRef.current) return;
          const customUniformLocation = gl.getUniformLocation(shaderProgramRef.current, name);
          if (!customUniformLocation) return;
          processUniform(
            gl,
            customUniformLocation,
            currentUniform.type as UniformType,
            currentUniform.value
          );
        }
      }
    }
    if (uniformsRef.current.iMouse?.isNeeded) {
      const mouseUniform = gl.getUniformLocation(shaderProgramRef.current, UNIFORM_MOUSE);
      gl.uniform4fv(mouseUniform, uniformsRef.current.iMouse.value as number[]);
    }
    if (uniformsRef.current.iChannelResolution?.isNeeded) {
      const channelResUniform = gl.getUniformLocation(
        shaderProgramRef.current,
        UNIFORM_CHANNELRESOLUTION
      );
      gl.uniform3fv(channelResUniform, uniformsRef.current.iChannelResolution.value as number[]);
    }
    if (uniformsRef.current.iDeviceOrientation?.isNeeded) {
      const deviceOrientationUniform = gl.getUniformLocation(
        shaderProgramRef.current,
        UNIFORM_DEVICEORIENTATION
      );
      gl.uniform4fv(
        deviceOrientationUniform,
        uniformsRef.current.iDeviceOrientation.value as number[]
      );
    }
    if (uniformsRef.current.iTime?.isNeeded) {
      const timeUniform = gl.getUniformLocation(shaderProgramRef.current, UNIFORM_TIME);
      gl.uniform1f(timeUniform, (timerRef.current += delta));
    }
    if (uniformsRef.current.iTimeDelta?.isNeeded) {
      const timeDeltaUniform = gl.getUniformLocation(shaderProgramRef.current, UNIFORM_TIMEDELTA);
      gl.uniform1f(timeDeltaUniform, delta);
    }
    if (uniformsRef.current.iDate?.isNeeded) {
      const d = new Date();
      const month = d.getMonth() + 1;
      const day = d.getDate();
      const year = d.getFullYear();
      const time =
        d.getHours() * 60 * 60 + d.getMinutes() * 60 + d.getSeconds() + d.getMilliseconds() * 0.001;
      const dateUniform = gl.getUniformLocation(shaderProgramRef.current, UNIFORM_DATE);
      gl.uniform4fv(dateUniform, [year, month, day, time]);
    }
    if (uniformsRef.current.iFrame?.isNeeded) {
      const timeDeltaUniform = gl.getUniformLocation(shaderProgramRef.current, UNIFORM_FRAME);
      gl.uniform1i(timeDeltaUniform, (uniformsRef.current.iFrame.value as number)++);
    }
    if (texturesArrRef.current.length > 0) {
      for (let index = 0; index < texturesArrRef.current.length; index++) {
        const texture = texturesArrRef.current[index];
        if (!texture) return;
        const { isVideo, _webglTexture, source, flipY, isLoaded } = texture;
        if (!isLoaded || !_webglTexture || !source) return;
        if (uniformsRef.current[`iChannel${index}`]?.isNeeded) {
          if (!shaderProgramRef.current) return;
          const iChannel = gl.getUniformLocation(shaderProgramRef.current, `iChannel${index}`);
          gl.activeTexture(gl.TEXTURE0 + index);
          gl.bindTexture(gl.TEXTURE_2D, _webglTexture);
          gl.uniform1i(iChannel, index);
          if (isVideo) {
            texture.updateTexture(_webglTexture, source as HTMLVideoElement, flipY);
          }
        }
      }
    }
  };

  const drawScene = (timestamp: number) => {
    const gl = glRef.current;
    if (!gl) return;
    gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.bindBuffer(gl.ARRAY_BUFFER, squareVerticesBufferRef.current);
    gl.vertexAttribPointer(vertexPositionAttributeRef.current ?? 0, 3, gl.FLOAT, false, 0, 0);
    setUniforms(timestamp);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    const mouseValue = uniformsRef.current.iMouse?.value;
    if (uniformsRef.current.iMouse?.isNeeded && lerp !== 1 && Array.isArray(mouseValue)) {
      const currentX = mouseValue[0] ?? 0;
      const currentY = mouseValue[1] ?? 0;
      mouseValue[0] = lerpVal(currentX, lastMouseArrRef.current[0] ?? 0, lerp);
      mouseValue[1] = lerpVal(currentY, lastMouseArrRef.current[1] ?? 0, lerp);
    }
    animFrameIdRef.current = requestAnimationFrame(drawScene);
  };

  const addEventListeners = () => {
    const options = { passive: true };
    if (uniformsRef.current.iMouse?.isNeeded && canvasRef.current) {
      canvasRef.current.addEventListener('mousemove', mouseMove, options);
      canvasRef.current.addEventListener('mouseout', mouseUp, options);
      canvasRef.current.addEventListener('mouseup', mouseUp, options);
      canvasRef.current.addEventListener('mousedown', mouseDown, options);
      canvasRef.current.addEventListener('touchmove', mouseMove, options);
      canvasRef.current.addEventListener('touchend', mouseUp, options);
      canvasRef.current.addEventListener('touchstart', mouseDown, options);
    }
    if (uniformsRef.current.iDeviceOrientation?.isNeeded) {
      window.addEventListener('deviceorientation', onDeviceOrientationChange, options);
    }
    if (canvasRef.current) {
      resizeObserverRef.current = new ResizeObserver(onResize);
      resizeObserverRef.current.observe(canvasRef.current);
      window.addEventListener('resize', onResize, options);
    }
  };

  const removeEventListeners = () => {
    const options = { passive: true } as EventListenerOptions;
    if (uniformsRef.current.iMouse?.isNeeded && canvasRef.current) {
      canvasRef.current.removeEventListener('mousemove', mouseMove, options);
      canvasRef.current.removeEventListener('mouseout', mouseUp, options);
      canvasRef.current.removeEventListener('mouseup', mouseUp, options);
      canvasRef.current.removeEventListener('mousedown', mouseDown, options);
      canvasRef.current.removeEventListener('touchmove', mouseMove, options);
      canvasRef.current.removeEventListener('touchend', mouseUp, options);
      canvasRef.current.removeEventListener('touchstart', mouseDown, options);
    }
    if (uniformsRef.current.iDeviceOrientation?.isNeeded) {
      window.removeEventListener('deviceorientation', onDeviceOrientationChange, options);
    }
    if (resizeObserverRef.current) {
      resizeObserverRef.current.disconnect();
      window.removeEventListener('resize', onResize, options);
    }
  };

  useEffect(() => {
    propsUniformsRef.current = propUniforms;
  }, [propUniforms]);

  // Main effect for initialization and cleanup
  useEffect(() => {
    const textures = texturesArrRef.current;

    function init() {
      initWebGL();
      const gl = glRef.current;
      if (gl && canvasRef.current) {
        gl.clearColor(...clearColor);
        gl.clearDepth(1.0);
        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);
        gl.viewport(0, 0, canvasRef.current.width, canvasRef.current.height);
        canvasRef.current.height = canvasRef.current.clientHeight;
        canvasRef.current.width = canvasRef.current.clientWidth;
        processCustomUniforms();
        processTextures();
        initShaders(preProcessFragment(fs || BASIC_FS), vs || BASIC_VS);
        initBuffers();
        requestAnimationFrame(drawScene);
        addEventListeners();
        onResize();
      }
    }

    requestAnimationFrame(init);

    // Cleanup function
    return () => {
      const gl = glRef.current;
      if (gl) {
        gl.getExtension('WEBGL_lose_context')?.loseContext();
        gl.useProgram(null);
        gl.deleteProgram(shaderProgramRef.current ?? null);
        if (textures.length > 0) {
          for (const texture of textures as Texture[]) {
            gl.deleteTexture(texture._webglTexture);
          }
        }
        shaderProgramRef.current = null;
      }
      removeEventListeners();
      cancelAnimationFrame(animFrameIdRef.current ?? 0);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Empty dependency array to run only once on mount

  return (
    <canvas ref={canvasRef} style={{ height: '100%', width: '100%', ...style }} {...canvasProps} />
  );
}
