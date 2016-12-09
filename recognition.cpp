/*****************************************************************************
*   Face Recognition using Eigenfaces or Fisherfaces
******************************************************************************/


#include "recognition.h"     // Treinar o sistema de reconhecimento facial e reconhecimento de uma pessoa a partir de uma imagem.

#include "ImageUtils.h"

// Iniciar a formação dos rostos recolhidos.
// "FaceRecognizer.Eigenfaces": Eigenfaces, também referidos como PCA (Turk e Pentland, 1991).
// "FaceRecognizer.Fisherfaces": Fisherfaces, também referidos como LDA (Belhumeur et al, 1997).
// "FaceRecognizer.LBPH": local padrão binário histogramas (Ahonen et al, 2006).
Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm)
{
    Ptr<FaceRecognizer> model;

    cout << "Learning the collected faces using the [" << facerecAlgorithm << "] algorithm ..." << endl;

    // Verifique se o módulo "contrib" é carregado dinamicamente em tempo de execução.
    bool haveContribModule = initModule_contrib();
    if (!haveContribModule) {
        cerr << "ERROR: The 'contrib' module is needed for FaceRecognizer but has not been loaded into OpenCV!" << endl;
        exit(1);
    }

    // Use a nova classe FaceRecognizer no módulo "contrib" do OpenCV:
    model = Algorithm::create<FaceRecognizer>(facerecAlgorithm);
    if (model.empty()) {
        cerr << "ERROR: The FaceRecognizer algorithm [" << facerecAlgorithm << "] is not available in your version of OpenCV. Please update to OpenCV v2.4.1 or newer." << endl;
        exit(1);
    }

    // Faça o treinamento real dos rostos recolhidos. Pode demorar alguns segundos ou minutos, dependendo de entrada!
    model->train(preprocessedFaces, faceLabels);

    return model;
}

// Converte a linha ou coluna da matriz (matriz float) para uma imagem de 8 bits retangular que pode ser exibida ou salva.
// Escalas os valores a ser entre 0-255.
Mat getImageFrom1DFloatMat(const Mat matrixRow, int height)
{
    // Fazer uma imagem de forma retangular, em vez de uma única linha.
    Mat rectangularMat = matrixRow.reshape(1, height);
    // Escala os valores a ser entre 0 a 255 e armazená-los como uma imagem uchar 8-bit regular.
    Mat dst;
    normalize(rectangularMat, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    return dst;
}

// Mostra os dados de reconhecimento de face interna, para ajudar a depuração.
void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight)
{
    try {  // Cerque as chamadas OpenCV por um bloco try / catch para não falhar se alguns parâmetros do modelo não estão disponíveis.

        // Mostra a face média (média estatística para cada pixel das imagens coletadas).
        Mat averageFaceRow = model->get<Mat>("mean");
        printMatInfo(averageFaceRow, "averageFaceRow");
        // Converte a linha da matriz (matriz flutuador 1D) para uma imagem de 8 bits regular.
        Mat averageFace = getImageFrom1DFloatMat(averageFaceRow, faceHeight);
        printMatInfo(averageFace, "averageFace");
        imshow("averageFace", averageFace);

        // Pegando os eigenvectors
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        printMatInfo(eigenvectors, "eigenvectors");

        // Mostra as melhores 20 eigenfaces
        for (int i = 0; i < min(20, eigenvectors.cols); i++) {
            // Cria um vetor coluna de eigenvector #i.
            // Note que clone() garante que vai ser contínuo, para que possamos tratá-lo como um array, caso contrário não podemos remodelá-lo a um retângulo.
            // Note que a classe FaceRecognizer já nos dá L2 autovetores normalizados, de modo que não temos a normalizar-los nós mesmos.
            Mat eigenvectorColumn = eigenvectors.col(i).clone();
            Mat eigenface = getImageFrom1DFloatMat(eigenvectorColumn, faceHeight);

            imshow(format("Eigenface%d", i), eigenface);
        }

        // Pegando os eigenvectors
        Mat eigenvalues = model->get<Mat>("eigenvalues");
        printMat(eigenvalues, "eigenvalues");

        vector<Mat> projections = model->get<vector<Mat> >("projections");
        cout << "projections: " << projections.size() << endl;
        for (int i = 0; i < (int)projections.size(); i++) {
            printMat(projections[i], "projections");
        }


    } catch (cv::Exception e) {
        cout << "WARNING: Missing FaceRecognizer properties." << endl;
    }

}


// Gerar um rosto aproximadamente reconstruído por back-projecting pelos eigenvectors e eigenvalues do dado (pré-processado).
Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace)
{
    // Uma vez que só podemos reconstruir o rosto para alguns tipos de modelos FaceRecognizer (ou seja: Eigenfaces ou Fisherfaces),
    // Devemos cercar as chamadas OpenCV por um bloco try / catch para que não bata em outros modelos.
    try {

        // Obter alguns dados necessários a partir do modelo FaceRecognizer.
        Mat eigenvectors = model->get<Mat>("eigenvectors");
        Mat averageFaceRow = model->get<Mat>("mean");

        int faceHeight = preprocessedFace.rows;

       // Projetar a imagem de entrada para o subespaço PCA.
        Mat projection = subspaceProject(eigenvectors, averageFaceRow, preprocessedFace.reshape(1,1));

        // Gerar o rosto reconstruído volta do subespaço PCA.
        Mat reconstructionRow = subspaceReconstruct(eigenvectors, averageFaceRow, projection);
    
        // Converte a matriz de linha de flutuação para uma imagem de 8 bits regular. Note que nós
        // não devemos usar "getImageFrom1DFloatMat ()" porque não queremos normalizar
        // os dados, uma vez que já está na escala perfeita.

         // Fazer uma imagem de forma retangular, em vez de uma única linha.
        Mat reconstructionMat = reconstructionRow.reshape(1, faceHeight);
        // Converte os pixels de ponto flutuante para regular de 8 bits uchar pixels.
        Mat reconstructedFace = Mat(reconstructionMat.size(), CV_8U);
        reconstructionMat.convertTo(reconstructedFace, CV_8U, 1, 0);

        return reconstructedFace;

    } catch (cv::Exception e) {
        cout << "WARNING: Missing FaceRecognizer properties." << endl;
        return Mat();
    }
}


// Comparar duas imagens, obtendo o erro L2 (raiz quadrada da soma do erro ao quadrado).
double getSimilarity(const Mat A, const Mat B)
{
    if (A.rows > 0 && A.rows == B.rows && A.cols > 0 && A.cols == B.cols) {
        // Calcular o erro relativo L2 entre as duas imagens.
        double errorL2 = norm(A, B, CV_L2);
        // Converte para uma escala razoável, uma vez que erro L2 é resumida em todos os pixels da imagem.
        double similarity = errorL2 / (double)(A.rows * A.cols);
        return similarity;
    }
    else {
        cout << "WARNING: Images have a different size in 'getSimilarity()'." << endl;
        return 100000000.0;  // Return a bad value
    }
}
