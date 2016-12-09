/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************/

const char *facerecAlgorithm = "FaceRecognizer.Fisherfaces";
//const char *facerecAlgorithm = "FaceRecognizer.Eigenfaces";


// Define quão confiante o algoritmo de verificação de Rosto deve ser o de decidir se é uma pessoa desconhecida ou uma pessoa conhecida.
// Um valor mais ou menos em torno de 0,5 parece OK para Eigenfaces ou 0,7 para Fisherfaces, mas você pode querer ajustá-lo para o seu
// Condições, e se você usar um algoritmo Face Recognition diferente.
// Note que um valor limite superior significa aceitar mais caras como pessoas conhecidas,
// Considerando que os valores mais baixos significam mais caras serão classificados como "desconhecido".

const float UNKNOWN_PERSON_THRESHOLD = 0.7f;


// Cascade Classifier arquivos, usados para Face Detection.
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Detector olho melhor para os olhos abertos ou fechados.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Detector olho melhor para os olhos abertos ou fechados.
//const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Detector olho bom para os olhos abertos ou fechados.
//const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Detector olho bom para os olhos abertos ou fechados.
const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Detector olho básico apenas para os olhos abertos.
const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Detector olho básico para os olhos abertos se eles poderiam usar óculos.


// Definir as dimensões face desejada. Note-se que "getPreprocessedFace ()" irá retornar um rosto quadrado.
const int faceWidth = 70;
const int faceHeight = faceWidth;

// Tentar definir a resolução da câmera. Note que isto só funciona para algumas câmeras em
// Alguns computadores e apenas para alguns motoristas, por isso não contam com ele para o trabalho!
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

// Parâmetros que controlam a freqüência para manter novos rostos quando coletá-los. 
// Caso contrário, o conjunto de treinamento poderia olhar para semelhantes uns aos outros!
const double CHANGE_IN_IMAGE_FOR_COLLECTION = 0.3;      // Quanto à imagem facial deve mudar antes de coletar uma nova foto do seu rosto para o treinamento.
const double CHANGE_IN_SECONDS_FOR_COLLECTION = 1.0;    // Quanto tempo deve passar antes de coletar uma nova foto do seu rosto para o treinamento.
const char *windowName = "WebcamFaceRec";   // Nome mostrado na janela de GUI.
const int BORDER = 8;  // Fronteira entre elementos da interface gráfica para a borda da imagem.

const bool preprocessLeftAndRightSeparately = true;   // Preprocess esquerdo e lado direito do rosto em separado, caso em que há luz mais forte em um lado.

// Defina como true se você quiser ver muitas janelas sendo criada, mostrando várias informações de depuração. Defina para 0 caso contrário.
bool m_debug = false;


#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>


#include "opencv2/opencv.hpp"


#include "detectObject.h"      
#include "preprocessFace.h"    
#include "recognition.h"    

#include "ImageUtils.h"     

using namespace cv;
using namespace std;


#if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      


// Modo de execução para o programa de interface gráfica interativa baseada em Webcam.
enum MODES {MODE_STARTUP=0, MODE_DETECTION, MODE_COLLECT_FACES, MODE_TRAINING, MODE_RECOGNITION, MODE_DELETE_ALL,   MODE_END};
const char* MODE_NAMES[] = {"Startup", "Deteccao", "Coletando Rostos", "Treinando", "Reconhecimento", "Deletando todos", "ERROR!"};
MODES m_mode = MODE_STARTUP;

int m_selectedPerson = -1;
int m_numPersons = 0;
vector<int> m_latestFaces;

// Position of GUI buttons:
Rect m_rcBtnAdd;
Rect m_rcBtnDel;
Rect m_rcBtnDebug;
int m_gui_faces_left = -1;
int m_gui_faces_top = -1;



// C ++ funções de conversão entre números inteiros (ou flutuadores) para std::string.
template <typename T> string toString(T t)
{
    ostringstream out;
    out << t;
    return out.str();
}

template <typename T> T fromString(string t)
{
    T out;
    istringstream in(t);
    in >> out;
    return out;
}

// Carrega o rosto e um ou dois olhos classificadores XML detecção.
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // Carrega o arquivo xml cascata classificador de Detecção de Rosto.
    try {  
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;

    // Carrega o arquivo xml cascata classificador Detecção dos olhos.
    try {  
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;

    // Carrega o arquivo xml cascata classificador Detecção dos olhos.
    try {   
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        // Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
        //exit(1);
    }
    else
        cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}

// Obter acesso à webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Obter acesso à câmera padrão.
    try {  
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}


// Desenha texto em uma imagem. O padrão é texto top-justificada à esquerda, mas você pode dar x coords negativas para texto justificado à direita;
// E / ou coordenadas y negativos para o texto-base justificada.
// Retorna o rect delimitadora em torno do texto elaborado.
Rect drawString(Mat img, string text, Point coord, Scalar color, float fontScale = 0.6f, int thickness = 1, int fontFace = FONT_HERSHEY_COMPLEX)
{
    // Pega o tamanho do texto e da linha de base.
    int baseline=0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);
    baseline += thickness;

    // Ajuste as coordenadas para a esquerda / direita justificado ou superior / inferior justificado.
    if (coord.y >= 0) {
        // As coordenadas são para o canto superior esquerdo do texto a partir do canto superior esquerdo da imagem, então se move para baixo por uma linha.
        coord.y += textSize.height;
    }
    else {
        // As coordenadas são para o canto inferior esquerdo do texto a partir do canto inferior esquerdo da imagem, por isso venha de baixo para cima.
        coord.y += img.rows - baseline + 1;
    }
    // Torne-se justificados à direita, se desejar.
    if (coord.x < 0) {
        coord.x += img.cols - textSize.width + 1;
    }

    // Obter a caixa delimitadora em torno do texto.
    Rect boundingRect = Rect(coord.x, coord.y - textSize.height, textSize.width, baseline + textSize.height);

    // Desenha texto anti-aliasing.
    putText(img, text, coord, fontFace, fontScale, color, thickness, CV_AA);

    // Que o usuário saiba o quão grande o seu texto é, em caso de querer organizar as coisas.
    return boundingRect;
}

// Desenha um botão GUI na imagem, usando drawString ().
// Pode especificar um minWidth se você quiser vários botões para todas têm a mesma largura.
// Retorna o rect delimitadora ao redor do botão tirada, permitindo-lhe posicionar botões ao lado do outro.
Rect drawButton(Mat img, string text, Point coord, int minWidth = 0)
{
    int B = BORDER;
    Point textCoord = Point(coord.x + B, coord.y + B);
    // Obter a caixa delimitadora em torno do texto.
    Rect rcText = drawString(img, text, textCoord, CV_RGB(0,0,0));
    // Desenha um retângulo preenchido em torno do texto.
    Rect rcButton = Rect(rcText.x - B, rcText.y - B, rcText.width + 2*B, rcText.height + 2*B);
    // Definir uma largura mínima botão.
    if (rcButton.width < minWidth)
        rcButton.width = minWidth;
    // Faça um retângulo branco semi-transparente.
    Mat matButton = img(rcButton);
    matButton += CV_RGB(90, 90, 90);
    // Desenha uma borda branca não transparente.
    rectangle(img, rcButton, CV_RGB(200,200,200), 1, CV_AA);

    // Desenha o texto real que será exibido, utilizando anti-aliasing.
    drawString(img, text, textCoord, CV_RGB(10,55,20));

    return rcButton;
}

bool isPointInRect(const Point pt, const Rect rc)
{
    if (pt.x >= rc.x && pt.x <= (rc.x + rc.width - 1))
        if (pt.y >= rc.y && pt.y <= (rc.y + rc.height - 1))
            return true;

    return false;
}

// Manipulador de eventos do mouse. Chamado automaticamente pelo OpenCV quando o usuário clica na janela GUI.
void onMouse(int event, int x, int y, int, void*)
{
    // Nós só se preocupamos com os liques de botão esquerdo do mouse.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;

    // Verificar se o usuário clicar em um dos nossos botões da GUI.
    Point pt = Point(x,y);
    if (isPointInRect(pt, m_rcBtnAdd)) {
        cout << "User clicked [Add Person] button when numPersons was " << m_numPersons << endl;
        // Verificar se já existe uma pessoa sem quaisquer rostos treinados, em seguida, usar essa pessoa em seu lugar.
        // Isto pode ser verificado por ver se existe uma imagem sua em "últimos rostos coletados".
        if ((m_numPersons == 0) || (m_latestFaces[m_numPersons-1] >= 0)) {
            // Adiciona uma nova pessoa
            m_numPersons++;
            m_latestFaces.push_back(-1); // Allocate space for an extra person.
            cout << "Num Persons: " << m_numPersons << endl;
        }
        // Use a pessoa recém-adicionada. Também use a mais nova pessoa, mesmo que essa pessoa estava vazio.
        m_selectedPerson = m_numPersons - 1;
        m_mode = MODE_COLLECT_FACES;
    }
    else if (isPointInRect(pt, m_rcBtnDel)) {
        cout << "User clicked [Delete All] button." << endl;
        m_mode = MODE_DELETE_ALL;
    }
    else if (isPointInRect(pt, m_rcBtnDebug)) {
        cout << "User clicked [Debug] button." << endl;
        m_debug = !m_debug;
        cout << "Debug mode: " << m_debug << endl;
    }
    else {
        cout << "User clicked on the image" << endl;
        // Verificar se o usuário clicar em um dos rostos na lista.
        int clickedPerson = -1;
        for (int i=0; i<m_numPersons; i++) {
            if (m_gui_faces_top >= 0) {
                Rect rcFace = Rect(m_gui_faces_left, m_gui_faces_top + i * faceHeight, faceWidth, faceHeight);
                if (isPointInRect(pt, rcFace)) {
                    clickedPerson = i;
                    break;
                }
            }
        }
        // Altere a pessoa atual e colete mais fotos para eles.
        if (clickedPerson >= 0) {
            // Altere a pessoa atual e colete mais fotos para eles.
            m_selectedPerson = clickedPerson; // Use a pessoa recém-adicionada.
            m_mode = MODE_COLLECT_FACES;
        }
        // Caso contrário, o usuário clicou no centro.
        else {
            // Altere para o modo de treinamento se estava coletando rostos.
            if (m_mode == MODE_COLLECT_FACES) {
                cout << "User wants to begin training." << endl;
                m_mode = MODE_TRAINING;
            }
        }
    }
}


// Loop principal que corre para sempre, até que as batidas do usuário Escape para sair.
void recognizeAndTrainUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_prepreprocessedFace;
    double old_time = 0;

    // Uma vez que já está inicializada, vamos iniciar no modo de detecção.
    m_mode = MODE_DETECTION;


    // Roda para sempre, até o usuário apertar Escape para sair.
    while (true) {

        // Pega o próximo frame da câmera. Note que você não pode modificar os quadros da câmera.
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }

        // Obter uma cópia do frame da câmera que podemos tirar para.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

        // Executar o sistema de reconhecimento de rosto na imagem da câmera. Ele vai tirar algumas coisas para a imagem dada, por isso certifique-se que não é só de leitura de memória!
        int identity = -1;

        /// Encontre um rosto e pré-processe para que ele tenha um tamanho padrão e contraste e brilho.
        Rect faceRect;  
        Rect searchedLeftEye, searchedRightEye; 
        Point leftEye, rightEye;    /
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

        bool gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;

        // Desenha um retângulo com anti-aliasing em torno do rosto detectado.
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);

            // Desenha círculos anti-aliasing de luz azul para os dois olhos.
            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) {   
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
            }
            if (rightEye.x >= 0) { 
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
            }
        }

        if (m_mode == MODE_DETECTION) {
            // Não fazer nada de especial.
        }
        else if (m_mode == MODE_COLLECT_FACES) {
            // Verificar se foi detectado um rosto.
            if (gotFaceAndEyes) {

                // Verificar se esse rosto parece um pouco diferente dos rostos previamente coletados.
                double imageDiff = 10000000000.0;
                if (old_prepreprocessedFace.data) {
                    imageDiff = getSimilarity(preprocessedFace, old_prepreprocessedFace);
                }

                // Registra além disso, quando isso aconteceu.
                double current_time = (double)getTickCount();
                double timeDiff_seconds = (current_time - old_time)/getTickFrequency();

                // Apenas processar a face se é visivelmente diferente do quadro anterior e tem havido diferença de tempo visível.
                if ((imageDiff > CHANGE_IN_IMAGE_FOR_COLLECTION) && (timeDiff_seconds > CHANGE_IN_SECONDS_FOR_COLLECTION)) {
                    // Também adicionar a imagem de espelho para o conjunto de treinamento, por isso temos mais dados de treinamento, bem como para lidar com rostos olhando para a esquerda ou para a direita.
                    Mat mirroredFace;
                    flip(preprocessedFace, mirroredFace, 1);

                    // Adicione as imagens de rostos para a lista de rostos detectados.
                    preprocessedFaces.push_back(preprocessedFace);
                    preprocessedFaces.push_back(mirroredFace);
                    faceLabels.push_back(m_selectedPerson);
                    faceLabels.push_back(m_selectedPerson);

                    // Mantenha uma referência mais recente rosto de cada pessoa.
                    m_latestFaces[m_selectedPerson] = preprocessedFaces.size() - 2;  // Ponto para o rosto não espelhado.
                    // Mostra o número de rostos recolhidos. Mas uma vez também armazenar rostos espelhados, apenas mostrar quantas o usuário pensa que foi armazenado.
                    cout << "Saved face " << (preprocessedFaces.size()/2) << " for person " << m_selectedPerson << endl;

                    // Faça um flash branco no rosto, de modo que o usuário saiba a foto foi tirada.
                    Mat displayedFaceRegion = displayedFrame(faceRect);
                    displayedFaceRegion += CV_RGB(90,90,90);

                    // Mantenha uma cópia do rosto transformado, para comparar na próxima iteração.
                    old_prepreprocessedFace = preprocessedFace;
                    old_time = current_time;
                }
            }
        }
        else if (m_mode == MODE_TRAINING) {

            // Verificar se não há dados suficientes para treinar. Para Eigenfaces, podemos aprender apenas uma pessoa, se quisermos, mas para Fisherfaces,
             // Precisamos de pelo menos 2 pessoas caso contrário ele irá falhar!
            bool haveEnoughData = true;
            if (strcmp(facerecAlgorithm, "FaceRecognizer.Fisherfaces") == 0) {
                if ((m_numPersons < 2) || (m_numPersons == 2 && m_latestFaces[1] < 0) ) {
                    cout << "Warning: Fisherfaces needs atleast 2 people, otherwise there is nothing to differentiate! Collect more data ..." << endl;
                    haveEnoughData = false;
                }
            }
            if (m_numPersons < 1 || preprocessedFaces.size() <= 0 || preprocessedFaces.size() != faceLabels.size()) {
                cout << "Warning: Need some training data before it can be learnt! Collect more data ..." << endl;
                haveEnoughData = false;
            }

            if (haveEnoughData) {
                // Iniciar a formação dos rostos recolhidos usando Eigenfaces ou um algoritmo similar.
                model = learnCollectedFaces(preprocessedFaces, faceLabels, facerecAlgorithm);

                // Mostra os dados de reconhecimento de face interna, para ajudar a depuração.
                if (m_debug)
                    showTrainingDebugData(model, faceWidth, faceHeight);

                // Agora que o treinamento acabou, podemos começar a reconhecer!
                m_mode = MODE_RECOGNITION;
            }
            else {
                // Como não há dados de treinamento suficientes, volte para o modo de recolher rostos!
                m_mode = MODE_COLLECT_FACES;
            }

        }
        else if (m_mode == MODE_RECOGNITION) {
            if (gotFaceAndEyes && (preprocessedFaces.size() > 0) && (preprocessedFaces.size() == faceLabels.size())) {

                // Gerar uma aproximação rosto de volta projetando-os eigenvectors e eigenvalues.
                Mat reconstructedFace;
                reconstructedFace = reconstructFace(model, preprocessedFace);
                if (m_debug)
                    if (reconstructedFace.data)
                        imshow("reconstructedFace", reconstructedFace);

                // Verifique se o rosto reconstruído se parece com o rosto pré-processado, caso contrário, é provável que seja uma pessoa desconhecida.
                double similarity = getSimilarity(preprocessedFace, reconstructedFace);

                string outputStr;
                if (similarity < UNKNOWN_PERSON_THRESHOLD) {
                    // Identificar quem é a pessoa da imagem de rosto pré-processados.
                    identity = model->predict(preprocessedFace);
                    outputStr = toString(identity);
                }
                else {
                    // Uma vez que a confiança é baixa, assumir que é uma pessoa desconhecida.
                    outputStr = "Unknown";
                }
                cout << "Identity: " << outputStr << ". Similarity: " << similarity << endl;

                // Mostra o nível de confiança para o reconhecimento em meados do topo da tela.
                int cx = (displayedFrame.cols - faceWidth) / 2;
                Point ptBottomRight = Point(cx - 5, BORDER + faceHeight);
                Point ptTopLeft = Point(cx - 15, BORDER);
                // Desenha uma linha cinza mostra o limite para uma pessoa "desconhecida".
                Point ptThreshold = Point(ptTopLeft.x, ptBottomRight.y - (1.0 - UNKNOWN_PERSON_THRESHOLD) * faceHeight);
                rectangle(displayedFrame, ptThreshold, Point(ptBottomRight.x, ptThreshold.y), CV_RGB(200,200,200), 1, CV_AA);
                // Cortar o nível de confiança entre 0,0 a 1,0, para mostrar na barra.
                double confidenceRatio = 1.0 - min(max(similarity, 0.0), 1.0);
                Point ptConfidence = Point(ptTopLeft.x, ptBottomRight.y - confidenceRatio * faceHeight);
                // Mostra a barra de confiança azul-claro.
                rectangle(displayedFrame, ptConfidence, ptBottomRight, CV_RGB(0,255,255), CV_FILLED, CV_AA);
                // Mostra a fronteira cinzenta do bar.
                rectangle(displayedFrame, ptTopLeft, ptBottomRight, CV_RGB(200,200,200), 1, CV_AA);
            }
        }
        else if (m_mode == MODE_DELETE_ALL) {
            // Reinicie tudo!
            m_selectedPerson = -1;
            m_numPersons = 0;
            m_latestFaces.clear();
            preprocessedFaces.clear();
            faceLabels.clear();
            old_prepreprocessedFace = Mat();

            // Reinicie em modo de detecção.
            m_mode = MODE_DETECTION;
        }
        else {
            cerr << "ERROR: Invalid run mode " << m_mode << endl;
            exit(1);
        }

        
        // Mostra a ajuda, ao mesmo tempo, mostrando o número de rostos recolhidos. Desde que nós também coletamos rostos espelhados, devemos apenas
        // Dizer ao usuário quantos rostos que pensam que salva (ignorando os rostos espelhados), daí dividir por dois.
        string help;
        Rect rcHelp;
        if (m_mode == MODE_DETECTION)
            help = "Click em [Adicionar Pessoa] quando estiver pronto para coletar os rostos.";
        else if (m_mode == MODE_COLLECT_FACES)
            help = "Click anywhere to train from your " + toString(preprocessedFaces.size()/2) + " faces of " + toString(m_numPersons) + " people.";
        else if (m_mode == MODE_TRAINING)
            help = "Please wait while your " + toString(preprocessedFaces.size()/2) + " faces of " + toString(m_numPersons) + " people builds.";
        else if (m_mode == MODE_RECOGNITION)
            help = "Click people on the right to add more faces to them, or [Add Person] for someone new.";
        if (help.length() > 0) {
            // Desenha-lo com um fundo preto e, em seguida, novamente com um primeiro plano branco.
            // Desde fronteira pode ser 0 e precisamos de uma posição negativa, subtraia 2 da fronteira por isso é sempre negativo.
            float txtSize = 0.4;
            drawString(displayedFrame, help, Point(BORDER, -BORDER-2), CV_RGB(0,0,0), txtSize);  // Sombra preta.
            rcHelp = drawString(displayedFrame, help, Point(BORDER+1, -BORDER-1), CV_RGB(255,255,255), txtSize);  // Texto Branco.
        }

        // Mostra o modo atual.
        if (m_mode >= 0 && m_mode < MODE_END) {
            string modeStr = "MODE: " + string(MODE_NAMES[m_mode]);
            drawString(displayedFrame, modeStr, Point(BORDER, -BORDER-2 - rcHelp.height), CV_RGB(0,0,0));       // Sombra preta
            drawString(displayedFrame, modeStr, Point(BORDER+1, -BORDER-1 - rcHelp.height), CV_RGB(0,255,0)); // Texto verde
        }

        // Mostra a face preprocessed atual em parte superior central da tela.
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) {
            // Obter uma versão BGR do rosto, uma vez que a saída é BGR cor.
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
            // Pega o ROI de destino (e certifique-se que está dentro da imagem!).
            // min (m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            /// Copiar os pixels de src para dst.
            srcBGR.copyTo(dstROI);
        }
        // Desenha uma borda anti-aliasing em torno do rosto, mesmo que isso não é mostrado.
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200,200,200), 1, CV_AA);

        // Desenha os botões da GUI na imagem principal.
        m_rcBtnAdd = drawButton(displayedFrame, "Adicionar Pessoa", Point(BORDER, BORDER));
        m_rcBtnDel = drawButton(displayedFrame, "Deletar Todas", Point(m_rcBtnAdd.x, m_rcBtnAdd.y + m_rcBtnAdd.height), m_rcBtnAdd.width);
        m_rcBtnDebug = drawButton(displayedFrame, "Debug", Point(m_rcBtnDel.x, m_rcBtnDel.y + m_rcBtnDel.height), m_rcBtnAdd.width);

        // Mostra a face mais recente para cada uma das pessoas recolhidos, no lado direito do visor.
        m_gui_faces_left = displayedFrame.cols - BORDER - faceWidth;
        m_gui_faces_top = BORDER;
        for (int i=0; i<m_numPersons; i++) {
            int index = m_latestFaces[i];
            if (index >= 0 && index < (int)preprocessedFaces.size()) {
                Mat srcGray = preprocessedFaces[index];
                if (srcGray.data) {
                    // Obter uma versão BGR do rosto, uma vez que a saída é BGR cor.
                    Mat srcBGR = Mat(srcGray.size(), CV_8UC3);
                    cvtColor(srcGray, srcBGR, CV_GRAY2BGR);
                    // Pega o ROI de destino (e certifique-se que está dentro da imagem!).
                    int y = min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
                    Rect dstRC = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                    Mat dstROI = displayedFrame(dstRC);
                    // Copiar os pixels de src para dst.
                    srcBGR.copyTo(dstROI);
                }
            }
        }

        // Destaque a pessoa que está sendo coletado, usando um retângulo vermelho em torno de seu rosto.
        if (m_mode == MODE_COLLECT_FACES) {
            if (m_selectedPerson >= 0 && m_selectedPerson < m_numPersons) {
                int y = min(m_gui_faces_top + m_selectedPerson * faceHeight, displayedFrame.rows - faceHeight);
                Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
                rectangle(displayedFrame, rc, CV_RGB(255,0,0), 3, CV_AA);
            }
        }

        // Destaque a pessoa que tenha sido reconhecido, usando um retângulo verde em torno de seu rosto.
        if (identity >= 0 && identity < 1000) {
            int y = min(m_gui_faces_top + identity * faceHeight, displayedFrame.rows - faceHeight);
            Rect rc = Rect(m_gui_faces_left, y, faceWidth, faceHeight);
            rectangle(displayedFrame, rc, CV_RGB(0,255,0), 3, CV_AA);
        }

        // Mostra o quadro da câmera na tela.
        imshow(windowName, displayedFrame);

        // Se o usuário deseja que todos os dados de depuração, mostrar a eles!
        if (m_debug) {
            Mat face;
            if (faceRect.width > 0) {
                face = cameraFrame(faceRect);
                if (searchedLeftEye.width > 0 && searchedRightEye.width > 0) {
                    Mat topLeftOfFace = face(searchedLeftEye);
                    Mat topRightOfFace = face(searchedRightEye);
                    imshow("topLeftOfFace", topLeftOfFace);
                    imshow("topRightOfFace", topRightOfFace);
                }
            }

            if (!model.empty())
                showTrainingDebugData(model, faceWidth, faceHeight);
        }

        
        // IMPORTANTE: Espere por pelo menos 20 milissegundos, para que a imagem pode ser exibida na tela!
        // Verifica Além disso, se uma tecla foi pressionada na janela GUI. 
        char keypress = waitKey(20);  // Isso é necessário se você quiser ver alguma coisa!

        if (keypress == VK_ESCAPE) {   // Tecla Espaço
            // Quit the program!
            break;
        }

    }//fim while
}


int main(int argc, char *argv[])
{
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;

    cout << "Realtime face detection + face recognition from a webcam using LBP and Eigenfaces or Fisherfaces." << endl;
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

    // Carrega o rosto e um ou dois olhos classificadores XML detecção.
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);

    cout << endl;
    cout << "Hit 'Escape' in the GUI window to quit." << endl;

    // Permitir que o usuário especifique um número da câmera, uma vez que nem todos os computadores será o mesmo número da câmera.
    int cameraNumber = 0;   // Altere este se você quiser usar um dispositivo de câmera diferente.
    if (argc > 1) {
        cameraNumber = atoi(argv[1]);
    }

    // Obter acesso à webcam.
    initWebcam(videoCapture, cameraNumber);

    // Tentar definir a resolução da câmera. Note que isto só funciona para algumas câmeras em
    // Alguns computadores e apenas para alguns motoristas, por isso não contam com ele para o trabalho!
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Cria uma janela GUI para exibição na tela.
    namedWindow(windowName); // Janela redimensionável, pode não funcionar no Windows.
    // Obter OpenCV para chamar automaticamente a minha função "onmouse ()" quando o usuário clica na janela GUI.
    setMouseCallback(windowName, onMouse, 0);

    // Rode Face Recogintion interativamente da webcam. Esta função é executado até que o usuário saía.
    recognizeAndTrainUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

    return 0;
}
