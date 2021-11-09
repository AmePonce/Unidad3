//Datos de Configuración
//TODO: Juega con el número de neuronas de la capa oculta y la tasa de aprendizaje, observa que sucede
const NUM_NEURONAS_CAPA_OCULTA = 12;
const TASA_DE_APRENDIZAJE = 0.1;
var noHeParadoAntes = true;
var ciclosDeAprendizaje = 0;
//! datos para entrenar
    //!datos de entrada (XOR)
    const datosEntrada = [[0,0],[0,1],[1,0],[1,1]];

    //!resultados esperrados para cada dato de entrada
    const datosEsperados = [[0],[1],[1],[0]];

//inicializacion de tensosre, peso y bias(utilizamos los datos y los convertimos)
const tensorEntrada = tf.tensor2s(datosEntrada, [4,2]);
const tensorEsperado = tf.tensor2d(datosEsperados, [4,1]);

//las variables se registran en tensorflow como variables entrenable
const pesosCapaUno = tf.variable(inicializaPesos([2, NUM_NEURONAS_CAPA_OCULTA], 2));
const biasCapaUno = tf.variable(tf.scalar(0));
const pesosCapaDos = tf.variable(inicializaPesos([NUM_NEURONAS_CAPA_OCULTA,1],NUM_NEURONAS_CAPA_OCULTA));
const biasCapaDos  = tf.variable(tf.scalar(0));

//definicion del modelo de red neuronal con TensorFlow.js core API
//son funciones que toman uno o mas tensores y devulven el tensor
//dichas funciones utilizan tf.variables que son los parametros entrenables
function model(xs) {
    const hiddenLayer = tf.tidy( function(){
        //peso,bias y funcion RELU 
        return xs.matMul(pesosCapaUno).add(biasCapaUno).relu();
    });
    //pesos, bias y funcion Sigmoide
    retmodel = hiddenLayer.matMul(pesosCapaDos).add(biasCapaDos).sigmoid();
    return retmodel;
}
//Gradientes de descenso de 0.1
const optimizador = rf.train.sgd(TASA_DE_APRENDIZAJE);

//inicializacion aleatoria de los pesos
function inicializacion(shape, prevLayerSize) {
    return tf.randomNormal(shape).mul(tf.scalar(Math.sqrt(2.0 / prevLayerSize)));
}

//Creamos la funcion de costos (aunque tf tambien nos provee de varias)
//aqui usamos minimos cuadros
function calculaCosto(y,output) {
    return tf.squaredDifference(y,output).sum().sqrt();
}

//función de entrenamiento que de manera repetitiva optimiza los parámetros  de la función de costo
async function entrena(iteraciones) {
    const regresaCosto =true;
    let costo;
    for(let i =0; i< iteraciones; i++){
        costo = optimizador.minimize(function() {
            return calculaCosto(tensorEsperado, model(tensorEntrada));
        },regresaCosto);
        if(i%100 === 0){
            costods = costo.dataSync()
            document.getElementById('divEntrenamiento').innerHTML += 'Perdida['+i+']:';
            updateCiclosDeAprendizaje(ciclosDeAprendizaje);
            if(costods<0.6 && noHeParadoAntes){
                noHeParadoAntes=false;
                break;
            }
        }
        await tf.nextFrame();
        ciclosDeAprendizaje += 1;
    }

    updateCiclosDeAprendizaje(ciclosDeAprendizaje);
    return costo.dataSync();
}
function updateCiclosDeAprendizaje(ciclos){
    $("#divCliclosDeAprendizaje")[0].innerHTML="Ciclos de Aprendizaje:" + ciclos; 
}

async function aprendeXor() {
    const timeStart = perfonmance.now();
    const iteraciones = Math.floor(Math.random()*200+400);
    document.getElementById('divEntrenamiento').innerHTML += <br>Número de iteraciones</br>;
    const loss = await entrena(interaciones);
    const time = performance.now()- timeStart;
    document.getElementById('divEntrenamiento').innerHTML += '<br>perdida:'+loss[0]+'</br>';
    document.getElementById('divEntrenamiento').innerHTML += 'duracion del entrenamiento';   
}
async function pruebaXor() {
    var timeSart2 = 0
    var time2 = 0 
    var strresult = "";
    timeSart2 = performance.now();
    for (i=0; i=datosEntrada.length; i++){
        const inputData = tf.tensor2d([datosEntrada[i]],[1,2]);
        const expectedOutput = tf.tensor1d(datosEsperados[i]);
        const val = model(inputData);
        const myVal = await val.data()
        strresult =strresult + "(" + datosEntrada[i][0]+","+ datosEntrada[i][0];
    }
    time2 = performance.now()-timeSart2;
    document.getElementById('divPrueva').innerHTML = '<br>Duracion de la prueba:'+timeSart2;
    document.getElementById('divPrueva').innerHTML += strresult;
    document.getElementById('divPrueva').innerHTML += "<br>Error:"+ calculaCosto(timeSart2);
}
function imprimePEsosYBias(){
    console.log(pesosCapaUno.dataSync());
    console.log(biasCapaUno.dataSync());
    console.log(pesosCapaDos.dataSync());
    console.log(biasCapaUno.dataSync());
}