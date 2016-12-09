/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************/

const double DESIRED_LEFT_EYE_X = 0.16;     // Controla quantos rostos serão exibidos depois do processamento
const double DESIRED_LEFT_EYE_Y = 0.14;
const double FACE_ELLIPSE_CY = 0.40;
const double FACE_ELLIPSE_W = 0.50;         // Precisa ser pelo menos 0.5
const double FACE_ELLIPSE_H = 0.80;         // Controla quão alta serão as máscaras


#include "detectObject.h"       // Detecta face ou olhos (usando LBP or Haar Cascades).
#include "preprocessFace.h"     // Processa as imagens de rostos, para o reconhecimento facil

#include "ImageUtils.h"      // Funções úteis

void detectBothEyes(const Mat &face, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, Point &leftEye, Point &rightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{

    // Como padrão eye.xml ou eyeglasses.xml: Encontra ambos os olhos em cerca de 40% dos rostos detectados, mas não detecta os olhos fechados.
    const float EYE_SX = 0.16f;
    const float EYE_SY = 0.26f;
    const float EYE_SW = 0.30f;
    const float EYE_SH = 0.28f;

    int leftX = cvRound(face.cols * EYE_SX);
    int topY = cvRound(face.rows * EYE_SY);
    int widthX = cvRound(face.cols * EYE_SW);
    int heightY = cvRound(face.rows * EYE_SH);
    int rightX = cvRound(face.cols * (1.0-EYE_SX-EYE_SW) );  // Começa pelo canto do olho direito

    Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
    Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));
    Rect leftEyeRect, rightEyeRect;

    // Retorna a janela de pesquisa para o vistante, se desejar
    if (searchedLeftEye)
        *searchedLeftEye = Rect(leftX, topY, widthX, heightY);
    if (searchedRightEye)
        *searchedRightEye = Rect(rightX, topY, widthX, heightY);

    // Procura na região esquerda, e depois na direita usando o primeiro detector de olho
    detectLargestObject(topLeftOfFace, eyeCascade1, leftEyeRect, topLeftOfFace.cols);
    detectLargestObject(topRightOfFace, eyeCascade1, rightEyeRect, topRightOfFace.cols);

    // Se o olho não for detectado, tenta um classificador diferente
    if (leftEyeRect.width <= 0 && !eyeCascade2.empty()) {
        detectLargestObject(topLeftOfFace, eyeCascade2, leftEyeRect, topLeftOfFace.cols);
    }
    

    // Se o olho não for detectado, tenta um classificador diferentes
    if (rightEyeRect.width <= 0 && !eyeCascade2.empty()) {
        detectLargestObject(topRightOfFace, eyeCascade2, rightEyeRect, topRightOfFace.cols);
    }

    if (leftEyeRect.width > 0) {   // Verifica se o olho foi detectado
        leftEyeRect.x += leftX;    // Ajusta o retângulo do olho esquerdo porque a bordas do rosto foi removida
        leftEyeRect.y += topY;
        leftEye = Point(leftEyeRect.x + leftEyeRect.width/2, leftEyeRect.y + leftEyeRect.height/2);
    }
    else {
        leftEye = Point(-1, -1);    // Retorna um ponto inválido
    }

    if (rightEyeRect.width > 0) { // Verifica se o olho foi detectado
        rightEyeRect.x += rightX; // Ajusta o retângulo do olho direito porque a bordas do rosto foi removida 
        rightEyeRect.y += topY;  
        rightEye = Point(rightEyeRect.x + rightEyeRect.width/2, rightEyeRect.y + rightEyeRect.height/2);
    }
    else {
        rightEye = Point(-1, -1);    // Retorna um ponto inválido
    }
}


// Equalizando separadamente para o lado esquerdo e direito do rosto.
void equalizeLeftAndRightHalves(Mat &faceImg)
{

    // É comum que há luz mais forte a partir de uma metade da face do que o outro. Nesse caso,
    // Se você simplesmente fazer a equalização de histograma em todo o rosto, então não faria metade escuro e
    // Metade brilhante. Então vamos fazer o histograma equalização separadamente em cada metade do rosto, então eles vão
    // ser um tanto semelhante, em média. Mas isso iria provocar uma borda afiada no meio da face, porque
    // A metade esquerda e metade direita seria de repente diferente. Então, nós também igualamos o histograma a toda
    // imagem, e na parte do meio que mistura as três imagens juntas para uma transição suave brilho.

    // É comum que existe uma luz forte 
    int w = faceImg.cols;
    int h = faceImg.rows;

    // 1) Primeiro, equalize todo o rosto
    Mat wholeFace;
    equalizeHist(faceImg, wholeFace);

    // 2) Equalize a metade esquerda e direita separadamente.
    int midX = w/2;
    Mat leftSide = faceImg(Rect(0,0, midX,h));
    Mat rightSide = faceImg(Rect(midX,0, w-midX,h));
    equalizeHist(leftSide, leftSide);
    equalizeHist(rightSide, rightSide);

    // 3) Combine a metada esquerda e direita com todo o rosto junto, de modo que ele tem uma transição suave.
    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            int v;
            if (x < w/4) {          // Esquerda 25%: Apenas usando o lado esquerdo do rosto
                v = leftSide.at<uchar>(y,x);
            }
            else if (x < w*2/4) {   // Metada esquerda: misturar a face esquerda e o rosto todo.
                int lv = leftSide.at<uchar>(y,x);
                int wv = wholeFace.at<uchar>(y,x);

                // Misture todo o rosto como ele se movesse mais para a direita ao longo da face.
                float f = (x - w*1/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * lv + (f) * wv);
            }
            else if (x < w*3/4) {   // Metada direita: misturar a face esquerda e o rosto todo.
                int rv = rightSide.at<uchar>(y,x-midX);
                int wv = wholeFace.at<uchar>(y,x);

                // Mistura mais do rosto do lado direito como ele se movesse mais para a direita ao longo da face.
                float f = (x - w*2/4) / (float)(w*0.25f);
                v = cvRound((1.0f - f) * wv + (f) * rv);
            }
            else {                  // Right 25%: just use the right face.
                v = rightSide.at<uchar>(y,x-midX);
            }
            faceImg.at<uchar>(y,x) = v;
        }// fim x loop
    }//fim y loop
}



// Cria uma imagem em tons de cinza do rosto que tem um tamanho padrão e contraste e brilho.
// "srcImg" deve ser uma cópia de todo o quadro da câmera de cor, de modo que possa tirar as posições de olho.
// Se 'doLeftAndRightSeparately' é verdade, então ele irá processar os lados esquerdo e direito separadamente,
// Modo que se houver uma forte luz sobre um dos lados, mas não a outra, ainda vai aparecer certo.
// Executa o pré-processamento de Rosto como uma combinação de:
// - Escala Geométrica, rotação e translação usando detecção dos olhos,
// - Suavizar os ruído de imagem usando um filtro Bilateral,
// - Uniformizar o brilho em ambos os lados esquerdo e direito do rosto independentemente separados usando a equalização de histograma,
// - Remoção de fundo e cabelo usando uma máscara elíptica.
// Retorna uma imagem quadrada rosto pré-processados ou NULL (ou seja: não conseguiu detectar o rosto e dois olhos).
// Se um rosto for encontrado, ele pode armazenar as coordenadas rect em 'storeFaceRect' e 'storeLeftEye' e 'storeRightEye',
// E regiões de busca de olho em 'searchedLeftEye' e 'searchedRightEye'.
Mat getPreprocessedFace(Mat &srcImg, int desiredFaceWidth, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2, bool doLeftAndRightSeparately, Rect *storeFaceRect, Point *storeLeftEye, Point *storeRightEye, Rect *searchedLeftEye, Rect *searchedRightEye)
{
    // Use rotos quadrados
    int desiredFaceHeight = desiredFaceWidth;

    // Marcando a detecção do rostos e as regiões de busca de olhos como inválida, se no caso elas não forem detectadas
    if (storeFaceRect)
        storeFaceRect->width = -1;
    if (storeLeftEye)
        storeLeftEye->x = -1;
    if (storeRightEye)
        storeRightEye->x= -1;
    if (searchedLeftEye)
        searchedLeftEye->width = -1;
    if (searchedRightEye)
        searchedRightEye->width = -1;


    // Acha o rosto mais largo
    Rect faceRect;
    detectLargestObject(srcImg, faceCascade, faceRect);

    // Verifica se o rosto foi detectado
    if (faceRect.width > 0) {


        // Devolve o rosto rect para o usuário se ele desejar
        if (storeFaceRect)
            *storeFaceRect = faceRect;

        Mat faceImg = srcImg(faceRect);    // Pega o rosto detectado

        // Se a imagem de entrada não está em escala de cinza, convertemos para BGR ou BGRA color para escala de cinza.
        Mat gray;
        if (faceImg.channels() == 3) {
            cvtColor(faceImg, gray, CV_BGR2GRAY);
        }
        else if (faceImg.channels() == 4) {
            cvtColor(faceImg, gray, CV_BGRA2GRAY);
        }
        else {
            // Acessa a imagem de entrada diretamente, desde que ela esteja em escala de cinza
            gray = faceImg;
        }


        // Procura pelos 2 olhos com a resolução inteira, porque a detecção de olhos precisa da máxima resolução possível
        Point leftEye, rightEye;
        detectBothEyes(gray, eyeCascade1, eyeCascade2, leftEye, rightEye, searchedLeftEye, searchedRightEye);

        // Devolve os olhos encontrados se o usuário desejar
        if (storeLeftEye)
            *storeLeftEye = leftEye;
        if (storeRightEye)
            *storeRightEye = rightEye;

        // Checa ambos os olhos forma detectados
        if (leftEye.x >= 0 && rightEye.x >= 0) {

            // Faz uma imagem do Rosto do mesmo tamanho que as imagens de treinamento.

            // Desde que encontramos ambos os olhos, isso permite girar, escalar e traduzir o rosto, para os dois olhos
            // Alinhar perfeitamente com as posições ideais dos olhos. Isso garante que os olhos estarão na horizontal,
            // e não muito distante para Esquerda OU Direita do Rosto, etc.

            // Pega o centro entre os dois olhos.
            Point2f eyesCenter = Point2f( (leftEye.x + rightEye.x) * 0.5f, (leftEye.y + rightEye.y) * 0.5f );
  
            // Pega o Ângulo entre os dois olhos.
            double dy = (rightEye.y - leftEye.y);
            double dx = (rightEye.x - leftEye.x);
            double len = sqrt(dx*dx + dy*dy);
            double angle = atan2(dy, dx) * 180.0/CV_PI; // Convertendo radiano para graus

            // Medições manuais mostraram que o centro do olho esquerdo deve ser idealmente em cerca de (0,19, 0,14) de uma imagem do rosto escalado.

            const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);

            // Obter a quantidade que precisamos para dimensionar a imagem para ser o tamanho fixo desejado que queremos.
            double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * desiredFaceWidth;
            double scale = desiredLen / len;
            // Obter a matriz de transformação para rotacionar e escalar a face ao ângulo e tamanho desejado.
            Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);
            // Deslocar o centro dos olhos para ser o centro desejado entre os olhos.
            rot_mat.at<double>(0, 2) += desiredFaceWidth * 0.5f - eyesCenter.x;
            rot_mat.at<double>(1, 2) += desiredFaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

            // Rotacionar, escalar e traduzir a imagem para o ângulo e tamanho e posição desejada!
            // Note-se que usamos "w" para a altura, em vez de 'h', porque a cara de entrada tem 1: 1 de relação de aspecto.
            Mat warped = Mat(desiredFaceHeight, desiredFaceWidth, CV_8U, Scalar(128)); // Limpar a imagem de saída para um cinza padrão.
            warpAffine(gray, warped, rot_mat, warped.size());
            
            // Dê um brilho a imagem padrão e contraste, no caso, era muito escuro ou tinham baixo contraste.
            if (!doLeftAndRightSeparately) {
                // Faça isso com todo o rosto
                equalizeHist(warped, warped);
            }
            else {
                // Faça-o separadamente para os lados esquerdo e direito do rosto.
                equalizeLeftAndRightHalves(warped);
            }
            

            // Use o "filtro Bilateral" para reduzir o ruído dos pixels para suavizar a imagem, mas mantendo as bordas afiadas na cara.
            Mat filtered = Mat(warped.size(), CV_8U);
            bilateralFilter(warped, filtered, 0, 20.0, 2.0);
           
            // Filtre os cantos do rosto, uma vez que, principalmente, só se preocupamos com as partes do meio.
            // Desenha uma elipse preenchida no meio da imagem de tamanho rosto.
            Mat mask = Mat(warped.size(), CV_8U, Scalar(0)); // Start with an empty mask.
            Point faceCenter = Point( desiredFaceWidth/2, cvRound(desiredFaceHeight * FACE_ELLIPSE_CY) );
            Size size = Size( cvRound(desiredFaceWidth * FACE_ELLIPSE_W), cvRound(desiredFaceHeight * FACE_ELLIPSE_H) );
            ellipse(mask, faceCenter, size, 0, 0, 360, Scalar(255), CV_FILLED);
          
            // Use a máscara para remover o pixels de fora
            Mat dstImg = Mat(warped.size(), CV_8U, Scalar(128)); // Clear the output image to a default gray.
           
            // Apply the elliptical mask on the face.
            // Aplicando a máscara eliptica sobre o rosto
            filtered.copyTo(dstImg, mask);  // Copia os pixels não mascarados de filtrada para dstImg.
            //imshow("dstImg", dstImg);

        
            return dstImg;
        }
        /*
        else {
            // Since no eyes were found, just do a generic image resize.
            resize(gray, tmpImg, Size(w,h));
        }
        */
    }
    return Mat();
}
