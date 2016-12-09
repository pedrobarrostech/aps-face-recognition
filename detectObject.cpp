#include "detectObject.h"

// Procurar por objetos, como rostos na imagem usando os parâmetros dados, armazenando o cv::Rects em 'objetcs'.
// Pode usar Haar cascades ou LBP cascades para detecção de rosto, ou mesmo olho, boca, ou a detecção de carro.
// A entrada é temporariamente reduzido para 'scaledWidth' para a detecção mais rápida, uma vez que 200 é o suficiente para encontrar rostos.
void detectObjectsCustom(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth, int flags, Size minFeatureSize, float searchScaleFactor, int minNeighbors)
{
    // Se a imagem de entrada não está em tons de cinza, em seguida, converter a imagem colorida BGR ou BGRA em tons de cinza.
    Mat gray;
    if (img.channels() == 3) {
        cvtColor(img, gray, CV_BGR2GRAY);
    }
    else if (img.channels() == 4) {
        cvtColor(img, gray, CV_BGRA2GRAY);
    }
    else {
        // Acesse a imagem de entrada diretamente, uma vez que já está em tons de cinza.
        gray = img;
    }

    // Possivelmente reduzir a imagem, para rodar muito mais rápido.
    Mat inputImg;
    float scale = img.cols / (float)scaledWidth;
    if (img.cols > scaledWidth) {
        // Encolher a imagem, mantendo a mesma proporção.
        int scaledHeight = cvRound(img.rows / scale);
        resize(gray, inputImg, Size(scaledWidth, scaledHeight));
    }
    else {
        // Acesse a imagem de entrada diretamente, uma vez que já é pequena.
        inputImg = gray;
    }

    // Padronizar o brilho e contraste para melhorar as imagens escuras.
    Mat equalizedImg;
    equalizeHist(inputImg, equalizedImg);

    // Detectar objetos na pequena imagem em tons de cinza.
    cascade.detectMultiScale(equalizedImg, objects, searchScaleFactor, minNeighbors, flags, minFeatureSize);

    // Aumentar os resultados se a imagem foi temporariamente reduzido antes da detecção.
    if (img.cols > scaledWidth) {
        for (int i = 0; i < (int)objects.size(); i++ ) {
            objects[i].x = cvRound(objects[i].x * scale);
            objects[i].y = cvRound(objects[i].y * scale);
            objects[i].width = cvRound(objects[i].width * scale);
            objects[i].height = cvRound(objects[i].height * scale);
        }
    }

    // Aumentar os resultados se a imagem foi temporariamente reduzida antes da detecção.
    for (int i = 0; i < (int)objects.size(); i++ ) {
        if (objects[i].x < 0)
            objects[i].x = 0;
        if (objects[i].y < 0)
            objects[i].y = 0;
        if (objects[i].x + objects[i].width > img.cols)
            objects[i].x = img.cols - objects[i].width;
        if (objects[i].y + objects[i].height > img.rows)
            objects[i].y = img.rows - objects[i].height;
    }

    // Retorna com os retângulos de rosto detectados armazenados em "objetcs".
}


// Procurar por apenas um único objeto na imagem, tais como a maior cara, armazenando o resultado na 'largestObject'.
// Pode usar Haar cascades ou LBP cascades para detecção de rosto, ou mesmo olho, boca, ou a detecção de carro.
// A entrada é temporariamente reduzido para 'scaledWidth' para a detecção mais rápida, uma vez que 200 é o suficiente para encontrar rostos.
// Nota: detectLargestObject () deve ser mais rápido do que detectManyObjects ().
void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth)
{
    // Apenas busca para apenas um objeto (o maior na imagem).
    int flags = CASCADE_FIND_BIGGEST_OBJECT; // | CASCADE_DO_ROUGH_SEARCH;
    // Tamanho do menor objeto.
    Size minFeatureSize = Size(20, 20);
    // Como detalhado deve ser a busca. Deve ser maior do que 1,0.
    float searchScaleFactor = 1.1f;
    // Quanto as detecções devem ser filtradas. Isso deve depender de quão ruim são as falsas detecções são para o sistema.
    // MinNeighbors = 2 significa muito bom + más detecções e minNeighbors = 6 significa apenas boas detecções são dadas, mas alguns são perdidas.
    int minNeighbors = 4;

    // Execute objeto ou de Detecção de Rosto, procurando apenas um objeto (o maior na imagem).
    vector<Rect> objects;
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
    if (objects.size() > 0) {
        // Retorna o único objeto detectado.
        largestObject = (Rect)objects.at(0);
    }
    else {
        // Retorna um rect inválido.
        largestObject = Rect(-1,-1,-1,-1);
    }
}

// Procurar por apenas um único objeto na imagem, tais como a maior cara, armazenando o resultado na 'largestObject'.
// Pode usar Haar cascades ou LBP cascades para detecção de rosto, ou mesmo olho, boca, ou a detecção de carro.
// A entrada é temporariamente reduzido para 'scaledWidth' para a detecção mais rápida, uma vez que 200 é o suficiente para encontrar rostos.
// Nota: detectLargestObject () deve ser mais rápido do que detectManyObjects ().
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth)
{
    // Procura de muitos objetos em uma imagem.
    int flags = CASCADE_SCALE_IMAGE;

    // Tamanho do menor objeto.
    Size minFeatureSize = Size(20, 20);
    // Como detalhado deve ser a busca. Deve ser maior do que 1,0.
    float searchScaleFactor = 1.1f;
    // Quanto as detecções devem ser filtradas. Isso deve depender de quão ruim são as falsas detecções são para o sistema.
    // MinNeighbors = 2 significa muito bom + más detecções e minNeighbors = 6 significa apenas boas detecções são dadas, mas alguns são perdidas.
    int minNeighbors = 4;

    // Execute objeto ou a Detecção de Rosto, à procura de muitos objetos na imagem um.
    detectObjectsCustom(img, cascade, objects, scaledWidth, flags, minFeatureSize, searchScaleFactor, minNeighbors);
}
