#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Image {
    int x, y;
    double *red, *green, *blue;
};

void examineImg(struct Image *img);
int exists(char *filename);
double nextValue(FILE *stream);
char *substring(char *string, int position, int length);
void getImgSize(char *filename, struct Image *img);
double *txtToArray(char *filename, size_t size);
int getImage(char *base_directory, struct Image *img_decomposed);

int main(int argc, char *argv[]) {

    struct Image *img = malloc(sizeof(struct Image));
    if (!getImage("imgs/2/", img)) return 1;

    examineImg(img);

    return 0;
}

int getImage(char *base_directory, struct Image *img_decomposed) {
    /*
    Lee tres archivos distintos, todos .txt, cada uno de los cuales
    representa un canal de color de una imagen. Estos fueron previamente
    generados usando un script de python, y nada más son interpretados por
    el presente código.

    Todos los archivos deben residir en el mismo directorio, y deben llevar
    como nombre el canal que representan. Ejemplo:
    Directorio: imgs/1/
    Rutas desde el ejecutable hasta los archivos:
        + ./imgs/1/red.txt
        + ./imgs/1/blue.txt
        + ./imgs/1/green.txt

    Tras procesar los archivos se genera una estructura Image, la cual
    contiene los siguientes elementos:
        + int x: Tamaño de la imágen en pixeles, en la dimensión x.
        + int y: Tamaño de la imágen en pixeles, en la dimensión y.
        + double *red: Puntero al array lineal que contiene los valores
        de rojo para la imagen.
        + double *green: Puntero al array lineal que contiene los valores
        de verde para la imagen.
        + double *blue: Puntero al array lineal que contiene los valores
        de azul para la imagen.
     */

    char redfile[2048];
    strcpy(redfile, base_directory);
    strcat(redfile, "red.txt");
    if (!exists(redfile)) return 1;

    char greenfile[2048];
    strcpy(greenfile, base_directory);
    strcat(greenfile, "green.txt");
    if (!exists(greenfile)) return 1;

    char bluefile[2048];
    strcpy(bluefile, base_directory);
    strcat(bluefile, "blue.txt");
    if (!exists(bluefile)) return 1;

    getImgSize(redfile, img_decomposed);

    int size = img_decomposed->x * img_decomposed->y * sizeof(double);
    img_decomposed->red = txtToArray(redfile, size);
    img_decomposed->blue = txtToArray(bluefile, size);
    img_decomposed->green = txtToArray(greenfile, size);

    return 1;
}

void examineImg(struct Image *img) {
    /* Ayuda para debug, y comprobar que todo funcione correctamente */
    printf("X: %d | Y: %d\n", img->x, img->y);

    int last = img->x * img->y - 1;
    printf("red\n");
    printf("First: %f\nLast: %f\n", img->red[0], img->red[last]);

    printf("green\n");
    printf("First: %f\nLast: %f\n", img->green[0], img->green[last]);

    printf("blue\n");
    printf("First: %f\nLast: %f\n", img->blue[0], img->blue[last]);
}

int exists(char *filename) {
    /* Valida que exista la ruta al archivo recibido */
    FILE *f;
    f = fopen(filename, "r");

    if (f) {
        fclose(f);
        return 1;
    }

    printf("El archivo '%s' no pudo abrirse.\n", filename);
    return 0;

}

void getImgSize(char *filename, struct Image *img) {
    /* Obtiene las dimensiones X e Y de la imagen */
    FILE *stream = fopen(filename, "r");
    char c_tmp;

    img->x = 1;
    img->y = 0;
    while ((c_tmp = getc(stream)) != '\n') {
        if (c_tmp == ' ') img->x++;
    }
    rewind(stream);

    while ((c_tmp = getc(stream)) != EOF) {
        if (c_tmp == '\n') img->y++;
    }
    fclose(stream);
}

double *txtToArray(char *filename, size_t size) {
    /* Procesa un solo archivo y lo convierte en un arreglo lineal
     * que contenga los datos leídos */
    FILE *stream = fopen(filename, "r");
    int c_tmp;
    double *arr = (double *) malloc(size);

    int counter = 0;
    double tmp = 0;
    while (1) {
        if ((tmp = nextValue(stream)) >= 0){
            arr[counter++] = tmp;
        }
        else break;
        c_tmp = getc(stream);
    }

    printf("%d\n", counter);

    fclose(stream);

    return arr;
}

double nextValue(FILE *stream) {
    /* Obtiene el siguiente valor inmediato del archivo.
     * Lee desde el primer caracter que sigue en el puntero de archivo,
     * y termina justa antes del espacio que separa valores.
     *
     * Como los archivos con los que se está trabajando siguen el mismo
     * formato, se utilizaron valores pre-codificados. Habrá que modificarlos
     * si cambia el formato del archivo, o hacerlo dinámico. */
    char n[26];
    char *mantissa_str;
    char *exponent_str;
    double mantissa;
    int exponent;
    int e_pos = 21;

    /* Llegamos a EOF */
    if (fgets(n, 25, stream) == NULL) {
        return -1;
    }

    mantissa_str = substring(n, 1, e_pos-1);
    exponent_str = substring(n, e_pos+1, 4);

    mantissa = atof(mantissa_str);
    exponent = atoi(exponent_str);

    if (exponent < 0) {
        int times = exponent * ((exponent > 0) - (exponent < 0));
        for (int i = 0; i < times; i++)
            mantissa = mantissa / 10;
    }
    return mantissa;
}

char *substring(char *string, int position, int length) {
    /* Obtiene una subcadena de <length> caracteres, empezando en
     * <position> */
    char *p;
    int c;

    p = malloc(length+1);

    if (p == NULL) {
        printf("No se pudo asignar memoria.\n");
        exit(1);
    }

    for (c = 0; c < length; c++) {
        *(p+c) = *(string+position-1);
        string++;
    }

    *(p+c) = '\0';

    return p;
}
