/*
 * Copyright 2016 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * The code is developed based on the original word2vec implementation from Google:
 * https://code.google.com/archive/p/word2vec/
 */

#include <cstring>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <stdio.h>
#ifdef USE_MKL
#include "mkl.h"
#include <clocale>
#include <wchar.h>
#include <stdlib.h>
#include <string.h>
#endif

using namespace std;

#define MAX_STRING 100
#define MAX_SUBSTRING = 100;
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
//Unicode range of Chinese characters
#define MIN_CHINESE 0x4E00
#define MAX_CHINESE 0x9FA5
typedef float real;
typedef unsigned int uint;
typedef unsigned long long ulonglong;

//struct vocab_word {
//    uint cn;
//    char *word;
//};
struct vocab_word {
    long long cn;
    int *point, character_size, *character_emb_select;
    unsigned int *character;
//    char *word, *code, codelen;
    char *word;
    /*
     * character[i]   : Unicode(the i-th character in the word) - MIN_CHINESE
     * character_size : the length of the word
                        (not equal to the length of string due to UTF-8 encoding)
     * character_emb_select[i][j] : count character[i] select the j-th cluster
     */
};
struct vocab_substring{
    char *substring;
};

class sequence {
public:
    int *indices;
    int *meta;
    int length;

    sequence(int len) {
        length = len;
        indices = (int *) _mm_malloc(length * sizeof(int), 64);
        meta = (int *) _mm_malloc(length * sizeof(int), 64);
    }

    ~sequence() {
        _mm_free(indices);
        _mm_free(meta);
    }
};

int binary = 0, debug_mode = 2;
bool disk = false;
int negative = 5, min_count = 5, num_threads = 12, min_reduce = 1, iter = 5, window = 5, batch_size = 11;
int vocab_max_size = 1000, vocab_size = 0, hidden_size = 100;
int vocab_substring_max_size = 1000, substring_size = 0;
ulonglong train_words = 0, file_size = 0;
real alpha = 0.025f, sample = 1e-3f;
const real EXP_RESOLUTION = EXP_TABLE_SIZE / (MAX_EXP * 2.0f);

char train_file[MAX_STRING], output_file[MAX_STRING], output_char[MAX_STRING];
char non_comp[MAX_STRING], char_init[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
const int table_size = 1e8;
int model_type = 0;
int cbow_type = 0;
struct vocab_word *vocab = NULL;
struct vocab_substring *substrings = NULL;
int *vocab_hash = NULL;
int *substring_hash = NULL;
int *table = NULL;
real *Wih = NULL, *Woh = NULL, *expTable = NULL;
real *WeightH = NULL;
real *charv;

//SEA
real *q;
real *k;
real *v;

long long character_size = 0;
int lang = 0;
void InitUnigramTable() {
    table = (int *) _mm_malloc(table_size * sizeof(int), 64);

    const real power = 0.75f;
    double train_words_pow = 0.;
    #pragma omp parallel for num_threads(num_threads) reduction(+: train_words_pow)
    for (int i = 0; i < vocab_size; i++) {
        train_words_pow += pow(vocab[i].cn, power);
    }

    int i = 0;
    real d1 = pow(vocab[i].cn, power) / train_words_pow;
    for (int a = 0; a < table_size; a++) {
        table[a] = i;
        if (a / (real) table_size > d1) {
            i++;
            d1 += pow(vocab[i].cn, power) / train_words_pow;
        }
        if (i >= vocab_size)
            i = vocab_size - 1;
    }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13)
            continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n')
                    ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *) "</s>");
                return;
            } else
                continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1)
            a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    uint hash = 0;
    for (int i = 0; i < strlen(word); i++)
        hash = hash * 257 + word[i];
    hash = hash % vocab_hash_size;
    return hash;
}
int GetSubstringHash(char *substring) {
    uint hash = 0;
    for (int i = 0; i< strlen(substring); i++)
        hash = hash * 257 + substring[i];
    hash = hash % vocab_hash_size;
    return hash;
}
// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1)
            return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word))
            return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}
int SearchSubstring(char *substring) {
    int hash = GetSubstringHash(substring);
    while (1) {
        if (substring_hash[hash] == -1)
            return -1;
        if (!strcmp(substring, substrings[substring_hash[hash]].substring))
            return substring_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}
// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin))
        return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddSubstringToVocab(char *substring) {
    int hash, length = strlen(substring) + 1;
    if (length > MAX_STRING)
        length = MAX_STRING;
    substrings[substring_size].substring = (char *) calloc(length, sizeof(char));
    strcpy(substrings[substring_size].substring, substring);
    substring_size++;
    // Reallocate memory if needed
    if (substring_size + 2 >= vocab_substring_max_size) {
        vocab_substring_max_size += 1000;
        substrings = (struct vocab_substring *) realloc(substrings, vocab_substring_max_size * sizeof(struct vocab_substring));
    }
    hash = GetSubstringHash(substring);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    substring_hash[hash] = substring_size - 1;
    return substring_size - 1;
}
int AddWordToVocab(char *word, int is_non_comp) {
    int hash, length = strlen(word) + 1;
    wchar_t wstr[MAX_STRING];
    uint len,i,pos;
    if (length > MAX_STRING)
        length = MAX_STRING;
    vocab[vocab_size].word = (char *) calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *) realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1)
        hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    if (!model_type || is_non_comp) return vocab_size - 1;
    //Chinese character
    if (model_type>=0 && lang==0){
        len = mbstowcs(wstr, word, MAX_STRING);
        for (i = 0; i < len; i++)
            if (wstr[i] < MIN_CHINESE || wstr[i] > MAX_CHINESE) {
                vocab[vocab_size - 1].character = 0;
                vocab[vocab_size - 1].character_size = 0;
                return vocab_size - 1;
            }
        vocab[vocab_size - 1].character =(unsigned int*) calloc(len, sizeof(unsigned int));
        vocab[vocab_size - 1].character_size = len;
        for (i = 0; i < len; i++) {
            //character
            vocab[vocab_size - 1].character[i] = wstr[i] - MIN_CHINESE;
        }
    }else if(model_type >= 0 && lang==1){
//        printf("%s,\t%d\n",word,length);
        char word_plus[20+2] = "<";
        strcat(word_plus,word);
        strcat(word_plus,">");
//        printf("%s\n",word_plus);
        int min_size = min(7,length-1);
        char ss[5];
        vocab[vocab_size - 1].character =(unsigned int*) calloc(min_size, sizeof(unsigned int));
        vocab[vocab_size - 1].character_size = min_size;
        for(int i = 0; i < min_size; i++){
            strncpy(ss,word_plus+i,4);
            ss[4] = '\0';
            int index = SearchSubstring(ss);
            if (index == -1) {
                index = AddSubstringToVocab(word);
            }
            vocab[vocab_size - 1].character[i] = index;
        }
    }
    return vocab_size - 1;
}
// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *) b)->cn - ((struct vocab_word *) a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    int size = vocab_size;
    train_words = 0;
    for (int i = 0; i < size; i++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if ((vocab[i].cn < min_count) && (i != 0)) {
            vocab_size--;
            if (vocab[i].character != NULL) free(vocab[i].character);
            free(vocab[i].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            int hash = GetWordHash(vocab[i].word);
            while (vocab_hash[hash] != -1)
                hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = i;
            train_words += vocab[i].cn;
        }
    }
    vocab = (struct vocab_word *) realloc(vocab, vocab_size * sizeof(struct vocab_word));
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int count = 0;
    for (int i = 0; i < vocab_size; i++) {
        if (vocab[i].cn > min_reduce) {
//            vocab[count].cn = vocab[i].cn;
//            vocab[count].word = vocab[i].word;
            memcpy(vocab + count, vocab + i, sizeof(struct vocab_word));
            count++;
        } else {
            if (vocab[i].character != NULL) free(vocab[i].character);
            free(vocab[i].word);
        }
    }
    vocab_size = count;
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));

    for (int i = 0; i < vocab_size; i++) {
        // Hash will be re-computed, as it is not actual
        int hash = GetWordHash(vocab[i].word);
        while (vocab_hash[hash] != -1)
            hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = i;
    }
    fflush(stdout);
    min_reduce++;
}
void LearnNonCompWord() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    fin = fopen(non_comp, "rb");
    if (fin == NULL) {
        printf("ERROR: non-compositional wordlist file not found!\n");
        exit(1);
    }
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word, 1);
            vocab[a].cn = 0;
        }
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
}
void LearnVocabFromTrainFile() {
    char word[MAX_STRING];

    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));
    memset(substring_hash, -1, vocab_hash_size * sizeof(int));
    FILE *fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }

    train_words = 0;
    vocab_size = 0;
    substring_size = 0;
    AddWordToVocab((char *) "</s>",0);
    if (strlen(non_comp)) LearnNonCompWord();
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        int i = SearchVocab(word);
        if (i == -1) {
            int a = AddWordToVocab(word,0);
            vocab[a].cn = 1;
        } else
            vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7)
            ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    FILE *fo = fopen(save_vocab_file, "wb");
    for (int i = 0; i < vocab_size; i++)
        fprintf(fo, "%s %d\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    memset(vocab_hash, -1, vocab_hash_size * sizeof(int));
    memset(substring_hash, -1, vocab_hash_size * sizeof(int));

    char c;
    vocab_size = 0;
    substring_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin))
            break;
        int i = AddWordToVocab(word,0);
        fscanf(fin, "%d%c", &vocab[i].cn, &c);
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %d\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fclose(fin);

    // get file size
    FILE *fin2 = fopen(train_file, "rb");
    if (fin2 == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin2, 0, SEEK_END);
    file_size = ftell(fin2);
    fclose(fin2);
}

void InitNet() {

    long long a, b, c, t1, t2;
    real *vec = (real*)calloc(hidden_size, sizeof(real)), len;
    wchar_t buf[10];
    FILE *file;
    if(lang == 1)
        character_size = substring_size;
    Wih = (real *) _mm_malloc(vocab_size * hidden_size * sizeof(real), 64);
    Woh = (real *) _mm_malloc(vocab_size * hidden_size * sizeof(real), 64);
    WeightH = (real *) _mm_malloc(vocab_size * sizeof(real),64);
    if (!Wih || !Woh || !WeightH) {
        printf("Memory allocation failed\n");
        exit(1);
    }

    #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
    for (int i = 0; i < vocab_size; i++) {
        memset(Wih + i * hidden_size, 0.f, hidden_size * sizeof(real));
        memset(Woh + i * hidden_size, 0.f, hidden_size * sizeof(real));
    }

    // initialization
    ulonglong next_random = 1;
    for (int i = 0; i < vocab_size * hidden_size; i++) {
        next_random = next_random * (ulonglong) 25214903917 + 11;
        Wih[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
    }

    if (model_type==1) {
        printf("char!\ncharactersize: %lld\n",character_size);
        a = posix_memalign((void **)&charv, 128, (long long)character_size * hidden_size * sizeof(real));
        if (charv == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < (long long)character_size * hidden_size; a++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            charv[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / hidden_size;
        }
    }else if(model_type==2){
        a = posix_memalign((void **)&charv, 128, (long long)character_size * hidden_size * sizeof(real));
        if (charv == NULL) {printf("Memory allocation failed\n"); exit(1);}
        for (a = 0; a < (long long)character_size * hidden_size; a++) {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            charv[a] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / hidden_size;
        }
        #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
        for (int i = 0; i < vocab_size; i++) {
            memset(WeightH + i , 0.f,  sizeof(real));
        }
        for (int i = 0; i < vocab_size; i++) {
            WeightH[i] = 0.5f;
        }
        printf("initNet()\n");
    }else if(model_type==3){
        q = (real *) _mm_malloc(hidden_size * hidden_size * sizeof(real), 64);
        k = (real *) _mm_malloc(hidden_size * hidden_size * sizeof(real), 64);
        v = (real *) _mm_malloc(hidden_size * hidden_size * sizeof(real), 64);


        if (!q || !k || !v) {
            printf("Memory allocation failed\n");
            exit(1);
        }

        #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
        for (int i = 0; i < hidden_size; i++) {
            memset(q + i * hidden_size, 0.f, hidden_size * sizeof(real));
            memset(k + i * hidden_size, 0.f, hidden_size * sizeof(real));
            memset(v + i * hidden_size, 0.f, hidden_size * sizeof(real));
        }


        // initialization
        ulonglong next_random = 1;
        for (int i = 0; i < hidden_size * hidden_size; i++) {
            next_random = next_random * (ulonglong) 25214903917 + 11;
            q[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
        }
        next_random = 2;
        for (int i = 0; i < hidden_size * hidden_size; i++) {
            next_random = next_random * (ulonglong) 25214903917 + 11;
            k[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
        }
        next_random = 3;
        for (int i = 0; i < hidden_size * hidden_size; i++) {
            next_random = next_random * (ulonglong) 25214903917 + 11;
            q[i] = (((next_random & 0xFFFF) / 65536.f) - 0.5f) / hidden_size;
        }
    }

}

ulonglong loadStream(FILE *fin, int *stream, const ulonglong total_words) {
    ulonglong word_count = 0;
    while (!feof(fin) && word_count < total_words) {
        int w = ReadWordIndex(fin);
        if (w == -1)
            continue;
        stream[word_count] = w;
        word_count++;
    }
    stream[word_count] = 0; // set the last word as "</s>"
    return word_count;
}

void Train_SGNS() {

#ifdef USE_MKL
    mkl_set_num_threads(1);
#endif

    if (read_vocab_file[0] != 0) {
        ReadVocab();
    }
    else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet();
    InitUnigramTable();

    real starting_alpha = alpha;
    ulonglong word_count_actual = 0;
    double start = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num();
        int local_iter = iter;
        ulonglong  next_random = id;
        ulonglong word_count = 0, last_word_count = 0;
        int sentence_length = 0, sentence_position = 0;
        int sen[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

        // load stream
        FILE *fin = fopen(train_file, "rb");
        fseek(fin, file_size * id / num_threads, SEEK_SET);

        ulonglong local_train_words = train_words / num_threads + (train_words % num_threads > 0 ? 1 : 0);
        int *stream;
        int w;

        if (!disk) {
            stream = (int *) _mm_malloc((local_train_words + 1) * sizeof(int), 64);
            local_train_words = loadStream(fin, stream, local_train_words);
            fclose(fin);
        }

        // temporary memory
        real * inputM = (real *) _mm_malloc(batch_size * hidden_size * sizeof(real), 64);
        real * outputM = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
        real * outputMd = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
        real * corrM = (real *) _mm_malloc((1 + negative) * batch_size * sizeof(real), 64);

        int inputs[2 * window + 1] __attribute__((aligned(64)));
        sequence outputs(1 + negative);

        #pragma omp barrier

        if (id == 0)
        {
            start = omp_get_wtime();
        }

        while (1) {
            if (word_count - last_word_count > 10000) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                last_word_count = word_count;
                if (debug_mode > 1) {
                    double now = omp_get_wtime();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk", 13, alpha,
                            word_count_actual / (real) (iter * train_words + 1) * 100,
                            word_count_actual / ((now - start) * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001f)
                    alpha = starting_alpha * 0.0001f;
            }
            if (sentence_length == 0) {
                while (1) {
                    if (disk) {
                        w = ReadWordIndex(fin);
                        if (feof(fin)) break;
                        if (w == -1) continue;
                    } else {
                        w = stream[word_count];
                    }
                    word_count++;
                    if (w == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ratio = (sample * train_words) / vocab[w].cn;
                        real ran = sqrtf(ratio) + ratio;
                        next_random = next_random * (ulonglong) 25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / 65536.f)
                            continue;
                    }
                    sen[sentence_length] = w;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }
            if ((disk && feof(fin)) || (word_count > local_train_words)) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                local_iter--;
                if (local_iter == 0) break;
                word_count = 0;
                last_word_count = 0;
                sentence_length = 0;
                if (disk) {
                    fseek(fin, file_size * id / num_threads, SEEK_SET);
                }
                continue;
            }

            int target = sen[sentence_position];
            outputs.indices[0] = target;
            outputs.meta[0] = 1;

            // get all input contexts around the target word
            next_random = next_random * (ulonglong) 25214903917 + 11;
            int b = next_random % window;

            int num_inputs = 0;
            for (int i = b; i < 2 * window + 1 - b; i++) {
                if (i != window) {
                    int c = sentence_position - window + i;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        break;
                    inputs[num_inputs] = sen[c];
                    num_inputs++;
                }
            }

            int num_batches = num_inputs / batch_size + ((num_inputs % batch_size > 0) ? 1 : 0);

            // start mini-batches
            for (int b = 0; b < num_batches; b++) {

                // generate negative samples for output layer
                int offset = 1;
                for (int k = 0; k < negative; k++) {
                    next_random = next_random * (ulonglong) 25214903917 + 11;
                    int sample = table[(next_random >> 16) % table_size];
                    if (!sample)
                        sample = next_random % (vocab_size - 1) + 1;
                    int* p = find(outputs.indices, outputs.indices + offset, sample);
                    if (p == outputs.indices + offset) {
                        outputs.indices[offset] = sample;
                        outputs.meta[offset] = 1;
                        offset++;
                    } else {
                        int idx = p - outputs.indices;
                        outputs.meta[idx]++;
                    }
                }
                outputs.meta[0] = 1;
                outputs.length = offset;

                // fetch input sub model
                int input_start = b * batch_size;
                int input_size  = min(batch_size, num_inputs - input_start);
                for (int i = 0; i < input_size; i++) {
                    memcpy(inputM + i * hidden_size, Wih + inputs[input_start + i] * hidden_size, hidden_size * sizeof(real));
                }
                // fetch output sub model
                int output_size = outputs.length;
                for (int i = 0; i < output_size; i++) {
                    memcpy(outputM + i * hidden_size, Woh + outputs.indices[i] * hidden_size, hidden_size * sizeof(real));
                }

#ifndef USE_MKL
                //output_size -> negative_size +1
                //meta -> label
                // input_size -> N
                // hidden_size -> D
                // f -> inn
                // g -> err*alpha
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];
                    for (int j = 0; j < input_size; j++) {
                        real f = 0.f, g;
                        #pragma simd
                        for (int k = 0; k < hidden_size; k++) {
                            f += outputM[i * hidden_size + k] * inputM[j * hidden_size + k];
                        }
                        int label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            g = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            g = label * alpha;
                        else
                            g = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        corrM[i * input_size + j] = g * c;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, input_size, hidden_size, 1.0f, outputM,
                        hidden_size, inputM, hidden_size, 0.0f, corrM, input_size);
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];
                    int offset = i * input_size;
                    #pragma simd
                    for (int j = 0; j < input_size; j++) {
                        real f = corrM[offset + j];
                        int label = (i ? 0 : 1);
                        if (f > MAX_EXP)
                            f = (label - 1) * alpha;
                        else if (f < -MAX_EXP)
                            f = label * alpha;
                        else
                            f = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                        corrM[offset + j] = f * c;
                    }
                }
#endif

#ifndef USE_MKL
                // outputMd -> update Mout
                for (int i = 0; i < output_size; i++) {
                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;
                        #pragma simd
                        for (int k = 0; k < input_size; k++) {
                            f += corrM[i * input_size + k] * inputM[k * hidden_size + j];
                        }
                        outputMd[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, hidden_size, input_size, 1.0f, corrM,
                        input_size, inputM, hidden_size, 0.0f, outputMd, hidden_size);
#endif

#ifndef USE_MKL

                //inputM -> Min
                for (int i = 0; i < input_size; i++) {
                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;
                        #pragma simd
                        for (int k = 0; k < output_size; k++) {
                            f += corrM[k * input_size + i] * outputM[k * hidden_size + j];
                        }
                        inputM[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, input_size, hidden_size, output_size, 1.0f, corrM,
                        input_size, outputM, hidden_size, 0.0f, inputM, hidden_size);
#endif

                // subnet update
                for (int i = 0; i < input_size; i++) {
                    int src = i * hidden_size;
                    int des = inputs[input_start + i] * hidden_size;
                    #pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Wih[des + j] += inputM[src + j];
                    }
                }

                for (int i = 0; i < output_size; i++) {
                    int src = i * hidden_size;
                    int des = outputs.indices[i] * hidden_size;
                    #pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Woh[des + j] += outputMd[src + j];
                    }
                }

            }

            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
        }
        _mm_free(inputM);
        _mm_free(outputM);
        _mm_free(outputMd);
        _mm_free(corrM);
        if (disk) {
            fclose(fin);
        } else {
            _mm_free(stream);
        }
    }
}

void Train_CBOWNS() {
#ifdef USE_MKL
    mkl_set_num_threads(1);
#endif
    if (read_vocab_file[0] != 0) {
        ReadVocab();
    } else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet(); //?
    InitUnigramTable();

    real starting_alpha = alpha; //learning rate
    ulonglong word_count_actual = 0; //current word number
    double start = 0;

    #pragma omp parallel num_threads(num_threads)
    {
        int id = omp_get_thread_num(); //thread id
        int local_iter = iter;
        ulonglong next_random = id;
        ulonglong word_count = 0, last_word_count = 0; //this thread word count
        int sentence_length = 0, sentence_position = 0; //sentence?
        int sen[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

        //load stream
        FILE *fin = fopen(train_file, "rb"); //open text file
        fseek(fin, file_size * id / num_threads, SEEK_SET); //get pointer
        //get how many words need be trained.
        ulonglong local_train_words = train_words / num_threads + (train_words % num_threads > 0 ? 1 : 0);
        int *stream;
        int w; //word

        if (!disk) {
            stream = (int *) _mm_malloc((local_train_words + 1) * sizeof(int), 64);
            local_train_words = loadStream(fin, stream, local_train_words); //read words
            fclose(fin);
        }

        //temporary memory for calculating
        real *inputM = (real *) _mm_malloc(batch_size * hidden_size * sizeof(real), 64);
        real *outputM = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
        real *outputMd = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
//        real * corrM = (real *) _mm_malloc((1 + negative) * batch_size * sizeof(real), 64);
        real *corrM = (real *) _mm_malloc((1 + negative) * sizeof(real), 64);
//        real * cbowM = (real *) _mm_malloc(hidden_size * sizeof(real),64);
        real cbowM[hidden_size] __attribute__((aligned(64)));
        int inputs[2 * window + 1] __attribute__((aligned(64))); //?
        sequence outputs(1 + negative);

        #pragma omp barrier

        if (id == 0) {
            start = omp_get_wtime();
        }

        while (1) {
            if (word_count - last_word_count > 10000) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                last_word_count = word_count;
                if (debug_mode > 1) {
                    double now = omp_get_wtime();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk", 13, alpha,
                           word_count_actual / (real) (iter * train_words + 1) * 100,
                           word_count_actual / ((now - start) * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001f)
                    alpha = starting_alpha * 0.0001f;
            }
            if (sentence_length == 0) {
                while (1) {
                    if (disk) {
                        w = ReadWordIndex(fin);
                        if (feof(fin)) break;
                        if (w == -1) continue;
                    } else {
                        w = stream[word_count];
                    }
                    word_count++;
                    if (w == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ratio = (sample * train_words) / vocab[w].cn;
                        real ran = sqrtf(ratio) + ratio;
                        next_random = next_random * (ulonglong) 25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / 65536.f)
                            continue;
                    }
                    sen[sentence_length] = w;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }
            if ((disk && feof(fin)) || (word_count > local_train_words)) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                local_iter--;
                if (local_iter == 0) break;
                word_count = 0;
                last_word_count = 0;
                sentence_length = 0;
                if (disk) {
                    fseek(fin, file_size * id / num_threads, SEEK_SET);
                }
                continue;
            }

            int target = sen[sentence_position];
            outputs.indices[0] = target;
            outputs.meta[0] = 1;

            // get all input contexts around the target word
            next_random = next_random * (ulonglong) 25214903917 + 11;
            int b = next_random % window;

            int num_inputs = 0;
            for (int i = b; i < 2 * window + 1 - b; i++) {
                if (i != window) {
                    int c = sentence_position - window + i;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        break;
                    inputs[num_inputs] = sen[c];
                    num_inputs++;
                }
            }

            int num_batches = num_inputs / batch_size + ((num_inputs % batch_size > 0) ? 1 : 0);

            // start mini-batches
            for (int b = 0; b < num_batches; b++) {
                //generate negative samples for output layer
                int offset = 1;
                for (int k = 0; k < negative; k++) {
                    next_random = next_random * (ulonglong) 25214903917 + 11;
                    int sample = table[(next_random >> 16) % table_size];
                    if (!sample)
                        sample = next_random % (vocab_size - 1) + 1;
                    int *p = find(outputs.indices, outputs.indices + offset, sample);
                    if (p == outputs.indices + offset) {
                        outputs.indices[offset] = sample;
                        outputs.meta[offset] = 1;
                        offset++;
                    } else {
                        int idx = p - outputs.indices;
                        outputs.meta[idx]++;
                    }
                }
                outputs.meta[0] = 1;
                outputs.length = offset;

                // fetch input sub model
                int input_start = b * batch_size;
                int input_size = min(batch_size, num_inputs - input_start);
                for (int i = 0; i < input_size; i++) {
                    memcpy(inputM + i * hidden_size, Wih + inputs[input_start + i] * hidden_size,
                           hidden_size * sizeof(real));
                }
                // fetch output sub model
                int output_size = outputs.length;
                for (int i = 0; i < output_size; i++) {
                    memcpy(outputM + i * hidden_size, Woh + outputs.indices[i] * hidden_size,
                           hidden_size * sizeof(real));
                }

#ifndef USE_MKL

                //calculate cbow average
                for (int k = 0; k < hidden_size; k++) cbowM[k] = 0.f;
                for (int j = 0; j < input_size; j++) {
                    #pragma simd
                    for (int k = 0; k < hidden_size; k++) {
                        cbowM[k] += inputM[j * hidden_size + k];
                    }
                }
                for (int k = 0; k < hidden_size; k++) cbowM[k] = cbowM[k] / input_size;

                //output_size -> negative_size +1
                //meta -> label
                // input_size -> N
                // hidden_size -> D
                // f -> inn
                // g -> err*alpha
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];

                    real f = 0.f, g;
                    #pragma simd
                    for (int k = 0; k < hidden_size; k++) {
                        f += outputM[i * hidden_size + k] * cbowM[k];
                    }
                    int label = (i ? 0 : 1);
                    if (f > MAX_EXP)
                        g = (label - 1) * alpha;
                    else if (f < -MAX_EXP)
                        g = label * alpha;
                    else
                        g = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                    corrM[i] = g * c;
                }
#else
                for (int k = 0; k < hidden_size; k++) cbowM[k] = 0.f;
                for (int j = 0; j < input_size; j++) {
                    #pragma simd
                    for (int k = 0; k < hidden_size; k++) {
                        cbowM[k] += inputM[j * hidden_size + k];
                    }
                }
                for (int k = 0; k < hidden_size; k++) cbowM[k] = cbowM[k] / input_size;

                //output_size -> negative_size +1
                //meta -> label
                // input_size -> N
                // hidden_size -> D
                // f -> inn
                // g -> err*alpha
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, 1, hidden_size, 1.0f, outputM,
                        hidden_size, cbowM, hidden_size, 0.0f, corrM, 1);
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];
                    real f = corrM[i];
                    int label = (i ? 0 : 1);
                    if (f > MAX_EXP)
                        f = (label - 1) * alpha;
                    else if (f < -MAX_EXP)
                        f = label * alpha;
                    else
                        f = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                    corrM[i] = f * c;

                }
#endif
#ifndef USE_MKL
            // outputMd -> update Mout
                for (int i = 0; i < output_size; i++) {
                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;
                        f += corrM[i] * cbowM[j];
                        outputMd[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, hidden_size, 1, 1.0f, corrM,
                        1, cbowM, hidden_size, 0.0f, outputMd, hidden_size);
#endif
#ifndef USE_MKL
                printf("use mkl\n");
            //inputM -> Min
                for (int j = 0; j < hidden_size; j++) {
                    real f = 0.f;
                    #pragma simd
                    for (int k = 0; k < output_size; k++) {
                        f += corrM[k] * outputM[k * hidden_size + j];
                    }
//                        inputM[i * hidden_size + j] = f / input_size;
                    cbowM[j] = f;
                }
#else
             //inputM -> Min
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, hidden_size, output_size, 1.0f, corrM,
                        1, outputM, hidden_size, 0.0f, cbowM, hidden_size);
#endif
                // subnet update
                for (int i = 0; i < input_size; i++) {
                    int src = i * hidden_size;
                    int des = inputs[input_start + i] * hidden_size;
                    #pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Wih[des + j] += cbowM[j];
                    }
                }

                for (int i = 0; i < output_size; i++) {
                    int src = i * hidden_size;
                    int des = outputs.indices[i] * hidden_size;
                    #pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Woh[des + j] += outputMd[src + j];
                    }
                }
            }
            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
        }
        _mm_free(inputM);
        _mm_free(outputM);
        _mm_free(outputMd);
        _mm_free(corrM);
        if (disk) {
            fclose(fin);
        } else {
            _mm_free(stream);
        }
    }

}

void MatrixAdd(real matrix0[],int row0,int col0,real matrix1[]){
    for(int r=0;r<row0;r++){
        for(int c=0;c<col0;c++){
            matrix0[r*col0+c] += matrix1[r*col0+c];
        }
    }
}
void VectorClear(real vector[], int size){
    for(int i = 0; i< size;i++){
        vector[i] = 0;
    }
}
void VectorAdd(real vector0[], int col0, real matrix1[], int matrix1_start, real alpha){
    for(int i = 0;i<col0;i++){
        vector0[i] += alpha * matrix1[matrix1_start*col0+i];
    }
}
void VectorAddBW(real vector1[], int col0, real matrix0[], int matrix0_start, real alpha){
    for(int i = 0;i<col0;i++){
        matrix0[matrix0_start*col0+i] += alpha * vector1[i];
    }
}
void VectorAddAlphaBW(real vector1[], int col0, real matrix0[], int matrix0_start, real alpha[],int index){
    for(int i = 0;i<col0;i++){
        alpha[index]+=matrix0[matrix0_start*col0+i]*vector1[i]/col0;
    }
}
void VectorMinus(real vector0[], int col0, real matrix1[], int matrix1_start, real alpha){
    for(int i = 0;i<col0;i++){
        vector0[i] -= alpha * matrix1[matrix1_start*col0+i];
    }
}
void VectorMinusBW(real vector1[], int col0, real matrix0[], int matrix0_start, real alpha){
    for(int i = 0;i<col0;i++){
        matrix0[matrix0_start*col0+i] -= alpha * vector1[i];
    }
}
void VectorMinusAlphaBW(real vector1[], int col0, real matrix0[], int matrix0_start, real alpha[],int index){
    for(int i = 0;i<col0;i++){
        alpha[index]-=matrix0[matrix0_start*col0+i]*vector1[i]/col0;
    }
}
void VectorMul(real matrix0[], int col0, real scale, int matrix0_start){
    for(int i = 0;i < col0;i++){
        matrix0[matrix0_start*col0+i] *= scale;
    }
}
void VectorMul(real vector0[], int col0, real matrix1[], int matrix0_start){
    for(int i = 0;i<col0;i++){
        vector0[i] *= matrix1[matrix0_start*col0+i];
    }
}
void Train_CBOWBasedNS() {

    int char_list_cnt;
    long long tot;
    wchar_t ch[10];
    char buf[10], pos;

    real *vec = (real*)calloc(hidden_size, sizeof(real));

    if (read_vocab_file[0] != 0) {
        ReadVocab();
    } else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet(); //?
    InitUnigramTable();

    real starting_alpha = alpha; //learning rate
    ulonglong word_count_actual = 0; //current word number
    double start = 0;

#pragma omp parallel num_threads(num_threads)
    {

        long long a, b, d, cw, t1, t2, word, last_word,  charv_id;
        long long l1, l2, c, label, index;
        long long *charv_id_list = (long long *)calloc(MAX_SENTENCE_LENGTH, sizeof(long long));
        int char_list_cnt;
        real *neu1char = (real*)calloc(hidden_size, sizeof(real));
        int id = omp_get_thread_num(); //thread id
        int local_iter = iter;
        ulonglong next_random = id;
        ulonglong word_count = 0, last_word_count = 0; //this thread word count
        int sentence_length = 0, sentence_position = 0; //sentence?
        int sen[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

        //load stream
        FILE *fin = fopen(train_file, "rb"); //open text file
        fseek(fin, file_size * id / num_threads, SEEK_SET); //get pointer
        //get how many words need be trained.
        ulonglong local_train_words = train_words / num_threads + (train_words % num_threads > 0 ? 1 : 0);
        int *stream;
        int w; //word

        if (!disk) {
            stream = (int *) _mm_malloc((local_train_words + 1) * sizeof(int), 64);
            local_train_words = loadStream(fin, stream, local_train_words); //read words
            fclose(fin);
        }

        //temporary memory for calculating
        real *inputM = (real *) _mm_malloc(batch_size * hidden_size * sizeof(real), 64);
        real *outputM = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
        real *outputMd = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
//        real * corrM = (real *) _mm_malloc((1 + negative) * batch_size * sizeof(real), 64);
        real *corrM = (real *) _mm_malloc((1 + negative) * sizeof(real), 64);
        real *weightM = (real *) _mm_malloc(batch_size * sizeof(real),64);
//        real * cbowM = (real *) _mm_malloc(hidden_size * sizeof(real),64);
        real cbowM[hidden_size] __attribute__((aligned(64)));
        int inputs[2 * window + 1] __attribute__((aligned(64))); //?
        sequence outputs(1 + negative);

        if(model_type==3){
            real *q_word;
            real *k_substrings;
            real *v_substrings;
            q_word = (real *) _mm_malloc(hidden_size*sizeof(real),64);
            k_substrings = (real *) _mm_malloc(MAX_SUBSTRING*hidden_size*sizeof(real),64);
            v_substrings = (real *) _mm_malloc(MAX_SUBSTRING*hidden_size*sizeof(real),64);
            real *Input_ss = (real *) _mm_malloc_(MAX_SUBSTRING * hidden_size * sizeof(real),64);
            real exp_att;
            exp_att = (real *) _mm_malloc(MAX_SUBSTRING * sizeof(real),64);
            if (!q_word || !k_substrings || !v_substrings || !Input_ss || !exp_att) {
                printf("Memory allocation failed\n");
                exit(1);
            }
        }


        #pragma omp barrier

        if (id == 0) {
            start = omp_get_wtime();
        }

        while (1) {
            if (word_count - last_word_count > 10000) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                last_word_count = word_count;
                if (debug_mode > 1) {
                    double now = omp_get_wtime();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk", 13, alpha,
                           word_count_actual / (real) (iter * train_words + 1) * 100,
                           word_count_actual / ((now - start) * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001f)
                    alpha = starting_alpha * 0.0001f;
            }
            if (sentence_length == 0) {
                while (1) {
                    if (disk) {
                        w = ReadWordIndex(fin);
                        if (feof(fin)) break;
                        if (w == -1) continue;
                    } else {
                        w = stream[word_count];
                    }
                    word_count++;
                    if (w == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ratio = (sample * train_words) / vocab[w].cn;
                        real ran = sqrtf(ratio) + ratio;
                        next_random = next_random * (ulonglong) 25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / 65536.f)
                            continue;
                    }
                    sen[sentence_length] = w;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }
            if ((disk && feof(fin)) || (word_count > local_train_words)) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                local_iter--;
                if (local_iter == 0) break;
                word_count = 0;
                last_word_count = 0;
                sentence_length = 0;
                if (disk) {
                    fseek(fin, file_size * id / num_threads, SEEK_SET);
                }
                continue;
            }

            int target = sen[sentence_position];
            outputs.indices[0] = target;
            outputs.meta[0] = 1;

            // get all input contexts around the target word
            next_random = next_random * (ulonglong) 25214903917 + 11;
            int b = next_random % window;

            int num_inputs = 0;
            cw = 0;
            char_list_cnt = 0;
            for (int i = b; i < 2 * window + 1 - b; i++) {
                if (i != window) {
                    int c = sentence_position - window + i;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        break;
                    inputs[num_inputs] = sen[c];
                    num_inputs++;
                }
            }

            int num_batches = num_inputs / batch_size + ((num_inputs % batch_size > 0) ? 1 : 0);

            // start mini-batches
            for (int b = 0; b < num_batches; b++) {
                //generate negative samples for output layer
                int offset = 1;
                for (int k = 0; k < negative; k++) {
                    next_random = next_random * (ulonglong) 25214903917 + 11;
                    int sample = table[(next_random >> 16) % table_size];
                    if (!sample)
                        sample = next_random % (vocab_size - 1) + 1;
                    int *p = find(outputs.indices, outputs.indices + offset, sample);
                    if (p == outputs.indices + offset) {
                        outputs.indices[offset] = sample;
                        outputs.meta[offset] = 1;
                        offset++;
                    } else {
                        int idx = p - outputs.indices;
                        outputs.meta[idx]++;
                    }
                }
                outputs.meta[0] = 1;
                outputs.length = offset;

                // fetch input sub model
                int input_start = b * batch_size;
                //debug
                if(input_start!=0){
                    printf("input: %d\n",input_start);
                    printf("input: %d\n",input_start);
                }
                int input_size = min(batch_size, num_inputs - input_start);
                for (int i = 0; i < input_size; i++) {
                    memcpy(inputM + i * hidden_size, Wih + inputs[input_start + i] * hidden_size,
                           hidden_size * sizeof(real));
                }
                // fetch output sub model
                int output_size = outputs.length;
                for (int i = 0; i < output_size; i++) {
                    memcpy(outputM + i * hidden_size, Woh + outputs.indices[i] * hidden_size,
                           hidden_size * sizeof(real));
                }
                if(model_type==2){
                    for (int i = 0; i < input_size; i++) {
                        memcpy(weightM + i , WeightH + inputs[input_start + i] ,
                               sizeof(real));
                    }
                }
                if(model_type==3){
                    memset(q_word,0.f,hidden_size*sizeof(real));
                    #pragma omp parallel for num_threads(num_threads) schedule(static, 1)
                    for(int i = 0; i < MAX_SUBSTRING; i++) {
                        memset(k_substings + i * hidden_size, 0.f, hidden_size * sizeof(real));
                        memset(v_substings + i * hidden_size, 0.f, hidden_size * sizeof(real));
                        memset(exp_att + i, 0.f, sizeof(real));
                    }

                }


                for (int k = 0; k < hidden_size; k++) cbowM[k] = 0.f;
//                matrixAdd()

                for (int j = 0; j < input_size; j++) {
                    if(model_type==0)
                        VectorAdd(cbowM,hidden_size,inputM,j,1.0f);
                    if(model_type==1){
                        last_word = inputs[j];
                        for (c = 0; c < hidden_size; c++) neu1char[c] = 0;
                        VectorAdd(neu1char,hidden_size,Wih,last_word,1.0f);
                        if (model_type && vocab[last_word].character_size) {
                            for (c = 0; c < vocab[last_word].character_size; c++) {
                                charv_id = vocab[last_word].character[c];
                                VectorAdd(neu1char,hidden_size,charv,charv_id,(1.0f / vocab[last_word].character_size));
                                charv_id_list[char_list_cnt] = charv_id;
                                char_list_cnt++;
                            }
                            VectorMul(neu1char,hidden_size,0.5f,0);
                        }
                        VectorAdd(cbowM,hidden_size,neu1char,0,1.0f);
                    }
                    if(model_type==2){
                        last_word = inputs[j];
                        for (c = 0; c < hidden_size; c++) neu1char[c] = 0;
                        VectorAdd(cbowM,hidden_size,Wih,last_word,weightM[j]);
                        if ( && vocab[last_word].character_size) {
                            for (c = 0; c < vocab[last_word].character_size; c++) {
                                charv_id = vocab[last_word].character[c];
                                VectorAdd(neu1char,hidden_size,charv,charv_id,((1.0f - weightM[j]) / vocab[last_word].character_size));
                                charv_id_list[char_list_cnt] = charv_id;
                                char_list_cnt++;
                            }
                        }

                        VectorAdd(cbowM,hidden_size,neu1char,0,1.0f);
                    }

                    // Attention forwad
                    if(model_type==3){
                        last_word = inputs[j];
                        int_t ss_size = vocab[last_word].character_size;
                        //TODO: q_word
                        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,1,hidden_size,hidden_size,
                                1.0f,Wih+hidden_size*last_word,hidden_size,q,hidden_size,q_word,hidden_size);
                        memcpy(Input_substring,Wih+hidden_size*last_word,hidden_size * sizeof(real));
                        for( int i = 1; i < ss_size;i++){
                            charv_id = vocab[last_word].character[i]
                            memcpy(Input_substring + i*hidden_size, charv + charv_id * hidden_size,
                                   hidden_size * sizeof(real));
                        }
                        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ss_size+1,hidden_size,hidden_size,
                                1.0f,Input_substring,hidden_size,k,hidden_size,k_substring,hidden_size);
                        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ss_size+1,hidden_size,hidden_size,
                                1.0f,Input_substring,hidden_size,v,hidden_size,v_substring,hidden_size);
                        real su = 0.f;
                        for( int i = 0; i < ss_size+1; i++){
                            su += exp_att[i];
                            exp_att[i] = exp(cblas_sdot(hidden_size,q_word,1,k_substring+i*hidden_size,1)/10);
                        }
                        for( int i = 0; i < ss_size+1; i++){
                            exp_att[i] /= su;
                        }
                        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,1,hidden_size,hidden_size,
                                1.0f,exp_att,hidden_size,v_substring,hidden_size,cbowM,hidden_size);
                    }

                }
                for (int k = 0; k < hidden_size; k++) cbowM[k] = cbowM[k] / input_size;

                //output_size -> negative_size +1
                //meta -> labelmodel_type
                // input_size -> N
                // hidden_size -> D
                // f -> inn
                // g -> err*alpha
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, 1, hidden_size, 1.0f, outputM,
                        hidden_size, cbowM, hidden_size, 0.0f, corrM, 1);
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];
                    real f = corrM[i];
                    int label = (i ? 0 : 1);
                    if (f > MAX_EXP)
                        f = (label - 1) * alpha;
                    else if (f < -MAX_EXP)
                        f = label * alpha;
                    else
                        f = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                    corrM[i] = f * c;

                }


                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, hidden_size, 1, 1.0f, corrM,
                        1, cbowM, hidden_size, 0.0f, outputMd, hidden_size);


                //inputM -> Min
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, hidden_size, output_size, 1.0f, corrM,
                        1, outputM, hidden_size, 0.0f, cbowM, hidden_size);

                // subnet update
                for (int i = 0; i < input_size; i++) {
//                    int src = i * hidden_size;
//                    int des = inputs[input_start + i] * hidden_size;

                    if(model_type==0)
                        VectorAddBW(cbowM,hidden_size,Wih,inputs[input_start + i], 1.0f);
                    if(model_type==1){
                        VectorAddBW(cbowM,hidden_size,Wih,inputs[input_start + i], 1.0f);
                        for(int j = 0;j<vocab[inputs[i]].character_size;j++){
                            VectorAddBW(cbowM,hidden_size,charv,vocab[inputs[i]].character[j],1.0f);
                        }
                    }
                    if(model_type==2){
                        double sum_a = 0.0;
                        for(int d = 0;d<hidden_size;d++){
//                            sum_a += inputM[i*hidden_size+d] * cbowM[d];
                            for(int c = 0;c<vocab[inputs[i]].character_size;c++){
                                sum_a-=(charv[d + vocab[inputs[i]].character[c] * hidden_size] * cbowM[d] ) / (vocab[inputs[i]].character_size);
                            }
                        }
                        if(sum_a > 10000000)
                            printf("%lld\n",sum_a);
                        WeightH[inputs[i]] += sum_a/hidden_size;
//                        VectorAddBW(cbowM,hidden_size,neu1char,0,1.0f);
                        VectorAddAlphaBW(cbowM,hidden_size,inputM,i,WeightH,inputs[i]);
//                        for(int c=0;c<vocab[inputs[i]].character_size;c++){
//                            VectorAddAlphaBW(neu1char,hidden_size,charv,vocab[inputs[i]].character[c],WeightH,inputs[i]);
//                        }
                        VectorAddBW(cbowM,hidden_size,Wih,inputs[i], weightM[i]);
                        for(int j = 0;j<vocab[inputs[i]].character_size;j++){
                            VectorAddBW(cbowM,hidden_size,charv,vocab[inputs[i]].character[j],(1.0f - weightM[i]) );
                        }
                    }
                }

                for (int i = 0; i < output_size; i++) {
                    int src = i * hidden_size;
                    int des = outputs.indices[i] * hidden_size;
                    #pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Woh[des + j] += outputMd[src + j];
                    }
                }
            }
            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
        }
        _mm_free(inputM);
        _mm_free(outputM);
        _mm_free(outputMd);
        _mm_free(corrM);
        if (disk) {
            fclose(fin);
        } else {
            _mm_free(stream);
        }
    }

}
void Train_CWENS() {

    int char_list_cnt;
    long long tot;
    wchar_t ch[10];
    char buf[10], pos;
    real *vec = (real*)calloc(hidden_size, sizeof(real));
#ifdef USE_MKL
    mkl_set_num_threads(1);
#endif
    if (read_vocab_file[0] != 0) {
        ReadVocab();
    } else {
        LearnVocabFromTrainFile();
    }
    if (save_vocab_file[0] != 0) SaveVocab();
    if (output_file[0] == 0) return;

    InitNet(); //?
    InitUnigramTable();

    real starting_alpha = alpha; //learning rate
    ulonglong word_count_actual = 0; //current word number
    double start = 0;

#pragma omp parallel num_threads(num_threads)
    {
        long long a, b, d, cw, t1, t2, word, last_word,  charv_id;
        long long l1, l2, c, label, index;
        long long *charv_id_list = (long long *)calloc(MAX_SENTENCE_LENGTH, sizeof(long long));
        int char_list_cnt;
        real *neu1char = (real*)calloc(hidden_size, sizeof(real));
        int id = omp_get_thread_num(); //thread id
        int local_iter = iter;
        ulonglong next_random = id;
        ulonglong word_count = 0, last_word_count = 0; //this thread word count
        int sentence_length = 0, sentence_position = 0; //sentence?
        int sen[MAX_SENTENCE_LENGTH] __attribute__((aligned(64)));

        //load stream
        FILE *fin = fopen(train_file, "rb"); //open text file
        fseek(fin, file_size * id / num_threads, SEEK_SET); //get pointer
        //get how many words need be trained.
        ulonglong local_train_words = train_words / num_threads + (train_words % num_threads > 0 ? 1 : 0);
        int *stream;
        int w; //word

        if (!disk) {
            stream = (int *) _mm_malloc((local_train_words + 1) * sizeof(int), 64);
            local_train_words = loadStream(fin, stream, local_train_words); //read words
            fclose(fin);
        }

        //temporary memory for calculating
        real *inputM = (real *) _mm_malloc(batch_size * hidden_size * sizeof(real), 64);
        real *outputM = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
        real *outputMd = (real *) _mm_malloc((1 + negative) * hidden_size * sizeof(real), 64);
//        real * corrM = (real *) _mm_malloc((1 + negative) * batch_size * sizeof(real), 64);
        real *corrM = (real *) _mm_malloc((1 + negative) * sizeof(real), 64);
//        real * cbowM = (real *) _mm_malloc(hidden_size * sizeof(real),64);
        real cbowM[hidden_size] __attribute__((aligned(64)));
        int inputs[2 * window + 1] __attribute__((aligned(64))); //?
        sequence outputs(1 + negative);

        #pragma omp barrier

        if (id == 0) {
            start = omp_get_wtime();
        }

        while (1) {
            if (word_count - last_word_count > 10000) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                last_word_count = word_count;
                if (debug_mode > 1) {
                    double now = omp_get_wtime();
                    printf("%cAlpha: %f  Progress: %.2f%%  Words/sec: %.2fk", 13, alpha,
                           word_count_actual / (real) (iter * train_words + 1) * 100,
                           word_count_actual / ((now - start) * 1000));
                    fflush(stdout);
                }
                alpha = starting_alpha * (1 - word_count_actual / (real) (iter * train_words + 1));
                if (alpha < starting_alpha * 0.0001f)
                    alpha = starting_alpha * 0.0001f;
            }
            if (sentence_length == 0) {
                while (1) {
                    if (disk) {
                        w = ReadWordIndex(fin);
                        if (feof(fin)) break;
                        if (w == -1) continue;
                    } else {
                        w = stream[word_count];
                    }
                    word_count++;
                    if (w == 0) break;
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        real ratio = (sample * train_words) / vocab[w].cn;
                        real ran = sqrtf(ratio) + ratio;
                        next_random = next_random * (ulonglong) 25214903917 + 11;
                        if (ran < (next_random & 0xFFFF) / 65536.f)
                            continue;
                    }
                    sen[sentence_length] = w;
                    sentence_length++;
                    if (sentence_length >= MAX_SENTENCE_LENGTH) break;
                }
                sentence_position = 0;
            }
            if ((disk && feof(fin)) || (word_count > local_train_words)) {
                ulonglong diff = word_count - last_word_count;
                #pragma omp atomic
                word_count_actual += diff;

                local_iter--;
                if (local_iter == 0) break;
                word_count = 0;
                last_word_count = 0;
                sentence_length = 0;
                if (disk) {
                    fseek(fin, file_size * id / num_threads, SEEK_SET);
                }
                continue;
            }

            int target = sen[sentence_position];
            outputs.indices[0] = target;
            outputs.meta[0] = 1;

            // get all input contexts around the target word
            next_random = next_random * (ulonglong) 25214903917 + 11;
            int b = next_random % window;

            int num_inputs = 0;
            cw = 0;
            char_list_cnt = 0;
            for (int i = b; i < 2 * window + 1 - b; i++) {
                if (i != window) {
                    int c = sentence_position - window + i;
                    if (c < 0)
                        continue;
                    if (c >= sentence_length)
                        break;
                    inputs[num_inputs] = sen[c];
                    num_inputs++;
                }
            }

            int num_batches = num_inputs / batch_size + ((num_inputs % batch_size > 0) ? 1 : 0);

            // start mini-batches
            for (int b = 0; b < num_batches; b++) {
                //generate negative samples for output layer
                int offset = 1;
                for (int k = 0; k < negative; k++) {
                    next_random = next_random * (ulonglong) 25214903917 + 11;
                    int sample = table[(next_random >> 16) % table_size];
                    if (!sample)
                        sample = next_random % (vocab_size - 1) + 1;
                    int *p = find(outputs.indices, outputs.indices + offset, sample);
                    if (p == outputs.indices + offset) {
                        outputs.indices[offset] = sample;
                        outputs.meta[offset] = 1;
                        offset++;
                    } else {
                        int idx = p - outputs.indices;
                        outputs.meta[idx]++;
                    }
                }
                outputs.meta[0] = 1;
                outputs.length = offset;

                // fetch input sub model
                int input_start = b * batch_size;
                int input_size = min(batch_size, num_inputs - input_start);
                for (int i = 0; i < input_size; i++) {
                    memcpy(inputM + i * hidden_size, Wih + inputs[input_start + i] * hidden_size,
                           hidden_size * sizeof(real));
                }
                // fetch output sub model
                int output_size = outputs.length;
                for (int i = 0; i < output_size; i++) {
                    memcpy(outputM + i * hidden_size, Woh + outputs.indices[i] * hidden_size,
                           hidden_size * sizeof(real));
                }
                #ifndef USE_MKL

                //calculate cbow average
                for (int k = 0; k < hidden_size; k++) cbowM[k] = 0.f;
                for (int j = 0; j < input_size; j++) {
                    last_word = inputs[j];
//                    printf("calc neu1char1\n");
                    for (c = 0; c < hidden_size; c++) neu1char[c] = 0;
                    for (c = 0; c < hidden_size; c++) neu1char[c] = Wih[c + last_word * hidden_size];
                    if (model_type && vocab[last_word].character_size) {
                        for (c = 0; c < vocab[last_word].character_size; c++) {
                            charv_id = vocab[last_word].character[c];
                            for (d = 0; d < hidden_size; d++)
                                neu1char[d] += charv[d + charv_id * hidden_size] / vocab[last_word].character_size;
                            charv_id_list[char_list_cnt] = charv_id;
                            char_list_cnt++;
                        }
                        for (d = 0; d < hidden_size; d++) neu1char[d] /= 2;
                    }
//                    printf("calc neu1char2\n");
                    #pragma simd
                    for (int k = 0; k < hidden_size; k++) {
                        cbowM[k] += neu1char[k];
                    }
//                    printf("neu->cbow2\n");
                }
                for (int k = 0; k < hidden_size; k++) cbowM[k] = cbowM[k] / input_size;

                //output_size -> negative_size +1
                //meta -> label
                // input_size -> N
                // hidden_size -> D
                // f -> inn
                // g -> err*alp a
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];

                    real f = 0.f, g;
                    #pragma simd
                    for (int k = 0; k < hidden_size; k++) {
                        f += outputM[i * hidden_size + k] * cbowM[k];
                    }
                    int label = (i ? 0 : 1);
                    if (f > MAX_EXP)
                        g = (label - 1) * alpha;
                    else if (f < -MAX_EXP)
                        g = label * alpha;
                    else
                        g = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                    corrM[i] = g * c;
                }
#else
//                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, input_size, hidden_size, 1.0f, outputM,
//                        hidden_size, inputM, hidden_size, 0.0f, corrM, input_size);
//                printf("CWE before calculate vec\n");
                for (int k = 0; k < hidden_size; k++) cbowM[k] = 0.f;
                for (int j = 0; j < input_size; j++) {
                    last_word = inputs[j];
                    for (c = 0; c < hidden_size; c++) neu1char[c] = 0;
                    for (c = 0; c < hidden_size; c++) neu1char[c] = Wih[c + last_word * hidden_size];
                    if (model_type && vocab[last_word].character_size) {
                        for (c = 0; c < vocab[last_word].character_size; c++) {
                            charv_id = vocab[last_word].character[c];
                            for (d = 0; d < hidden_size; d++)
                                neu1char[d] += charv[d + charv_id * hidden_size] / vocab[last_word].character_size;
                            charv_id_list[char_list_cnt] = charv_id;
                            char_list_cnt++;
                        }
                        for (d = 0; d < hidden_size; d++) neu1char[d] /= 2;
                    }

                    #pragma simd
                    for (int k = 0; k < hidden_size; k++) {
                        cbowM[k] += neu1char[k];
                    }
                }
//                printf("CWE before calculate vec\n");
                for (int k = 0; k < hidden_size; k++) cbowM[k] = cbowM[k] / input_size;

                //output_size -> negative_size +1
                //meta -> label
                // input_size -> N
                // hidden_size -> D
                // f -> inn
                // g -> err*alpha
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, output_size, 1, hidden_size, 1.0f, outputM,
                        hidden_size, cbowM, hidden_size, 0.0f, corrM, 1);
                for (int i = 0; i < output_size; i++) {
                    int c = outputs.meta[i];
                    #pragma simd
                    real f = corrM[i];
                    int label = (i ? 0 : 1);
                    if (f > MAX_EXP)
                        f = (label - 1) * alpha;
                    else if (f < -MAX_EXP)
                        f = label * alpha;
                    else
                        f = (label - expTable[(int) ((f + MAX_EXP) * EXP_RESOLUTION)]) * alpha;
                    corrM[i] = f * c;

                }
#endif
#ifndef USE_MKL
                // outputMd -> update Mout
                for (int i = 0; i < output_size; i++) {
                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;
                        f += corrM[i] * cbowM[j];
                        outputMd[i * hidden_size + j] = f;
                    }
                }
#else
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, output_size, hidden_size, 1, 1.0f, corrM,
                        1, cbowM, hidden_size, 0.0f, outputMd, hidden_size);
#endif
#ifndef USE_MKL
                printf("use mkl\n");
                //inputM -> Min
                for (int i = 0; i < input_size; i++) {
                    for (int j = 0; j < hidden_size; j++) {
                        real f = 0.f;
                        #pragma simd
                        for (int k = 0; k < output_size; k++) {
                            f += corrM[k] * outputM[k * hidden_size + j];
                        }
//                        inputM[i * hidden_size + j] = f / input_size;
                        inputM[i * hidden_size + j] = f;
                    }
                }
#else
                //inputM -> Min
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, 1, hidden_size, output_size, 1.0f, corrM,
                        1, outputM, hidden_size, 0.0f, cbowM, hidden_size);
#endif
                // subnet update
                for (int i = 0; i < input_size; i++) {
                    int src = i * hidden_size;
                    int des = inputs[input_start + i] * hidden_size;
                    #pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Wih[des + j] += cbowM[j];
                    }
//                    last_word = inputs[j];


                }
                //debug
//                printf("update charv1\n");
                for (c = 0; c < input_size; c++) {
//                    charv_id = charv_id_list[inputs[c]];
                    for(int i = 0;i<vocab[inputs[c]].character_size;i++){
                        for (d = 0; d < hidden_size; d++)
                            charv[d + vocab[inputs[c]].character[i] * hidden_size] += cbowM[d];
                    }
//
//                    charv_id = vocab[inputs[c]].cha
//                    for (d = 0; d < hidden_size; d++) charv[d + charv_id * hidden_size] += neu1e[d] * char_rate;
                }
//                printf("update charv2\n");

                for (int i = 0; i < output_size; i++) {
                    int src = i * hidden_size;
                    int des = outputs.indices[i] * hidden_size;
#pragma simd
                    for (int j = 0; j < hidden_size; j++) {
                        Woh[des + j] += outputMd[src + j];
                    }
                }
            }
            sentence_position++;
            if (sentence_position >= sentence_length) {
                sentence_length = 0;
            }
        }
        _mm_free(inputM);
        _mm_free(outputM);
        _mm_free(outputMd);
        _mm_free(corrM);
        if (disk) {
            fclose(fin);
        } else {
            _mm_free(stream);
        }
    }

}


int ArgPos(char *str, int argc, char **argv) {
    for (int a = 1; a < argc; a++)
        if (!strcmp(str, argv[a])) {
            //if (a == argc - 1) {
            //    printf("Argument missing for %s\n", str);
            //    exit(1);
            //}
            return a;
        }
    return -1;
}

void saveModel() {
    // save the model
    FILE *fo = fopen(output_file, "wb");
    // Save the word vectors
    fprintf(fo, "%d %d\n", vocab_size, hidden_size);
    for (int a = 0; a < vocab_size; a++) {
        fprintf(fo, "%s ", vocab[a].word);
        if (binary)
            for (int b = 0; b < hidden_size; b++)
                fwrite(&Wih[a * hidden_size + b], sizeof(real), 1, fo);
        else{
            for (int b = 0; b < hidden_size; b++){
                if(model_type==1){
                    real tmp = 0.f;
                    for(int i = 0;i<vocab[a].character_size;i++){
                        tmp += charv[vocab[a].character[i] * hidden_size+b];
                    }
                    tmp /= vocab[a].character_size;
                    fprintf(fo, "%f ", (Wih[a * hidden_size + b]+tmp)/2);
                }else if(model_type==2){
                    real tmp = 0.f;
                    for(int i = 0;i<vocab[a].character_size;i++){
                        tmp += charv[vocab[a].character[i] * hidden_size+b];
                    }
                    tmp /= vocab[a].character_size;
                    fprintf(fo, "%f ", (WeightH[a] * Wih[a * hidden_size + b])+(1-WeightH[a])*tmp);
                }else{
                    fprintf(fo, "%f ", Wih[a * hidden_size + b]);
                }

            }
        }
        fprintf(fo, "\n");
    }
    fclose(fo);
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "en_US.UTF-8");
    if (argc == 1) {
        printf("parallel word2vec (sgns) in shared memory system\n\n");
        printf("Options:\n");
        printf("Parameters for training:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-output <file>\n");
        printf("\t\tUse <file> to save the resulting word vectors\n");
        printf("\t-size <int>\n");
        printf("\t\tSet size of word vectors; default is 100\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <float>\n");
        printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
        printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
        printf("\t-negative <int>\n");
        printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
        printf("\t-threads <int>\n");
        printf("\t\tUse <int> threads (default 12)\n");
        printf("\t-iter <int>\n");
        printf("\t\tNumber of training iterations (default 5)\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-alpha <float>\n");
        printf("\t\tSet the starting learning rate; default is 0.025\n");

        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");

        printf("\t-cbow <int>\n");
        printf("\t\tSet cwe type; default is 1(CBOW), 0(SG)\n");
        printf("\t-model-type <int>\n");
        printf("\t\tSet cwe type; default is 1(CWE), 0(word2vec), 2(DSE), 3(SEA)\n");
        printf("\t-lang <int>\n");
        printf("\t\tSet language; default is 0(Chinese), 1(Other)\n");
        printf("\t-binary <int>\n");
        printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\t-batch-size <int>\n");
        printf("\t\tThe batch size used for mini-batch training; default is 11 (i.e., 2 * window + 1)\n");
        printf("\t-disk\n");
        printf("\t\tStream text from disk during training, otherwise the text will be loaded into memory before training\n");
        printf("\nExamples:\n");
        printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -binary 0 -iter 3\n\n");
        return 0;
    }

    output_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;

    int i;
    if ((i = ArgPos((char *) "-size", argc, argv)) > 0)
        hidden_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-train", argc, argv)) > 0)
        strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-save-vocab", argc, argv)) > 0)
        strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-read-vocab", argc, argv)) > 0)
        strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-debug", argc, argv)) > 0)
        debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0)
        cbow_type = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-model-type", argc, argv)) > 0)
        model_type = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-lang", argc, argv)) > 0)
        lang = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-binary", argc, argv)) > 0)
        binary = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-alpha", argc, argv)) > 0)
        alpha = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-output", argc, argv)) > 0)
        strcpy(output_file, argv[i + 1]);
    if ((i = ArgPos((char *) "-window", argc, argv)) > 0)
        window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-sample", argc, argv)) > 0)
        sample = atof(argv[i + 1]);
    if ((i = ArgPos((char *) "-negative", argc, argv)) > 0)
        negative = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-threads", argc, argv)) > 0)
        num_threads = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-iter", argc, argv)) > 0)
        iter = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-min-count", argc, argv)) > 0)
        min_count = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-batch-size", argc, argv)) > 0)
        batch_size = atoi(argv[i + 1]);
    if ((i = ArgPos((char *) "-disk", argc, argv)) > 0)
        disk = true;

    vocab = (struct vocab_word *) calloc(vocab_max_size, sizeof(struct vocab_word));
    substrings = (struct vocab_substring *) calloc(vocab_substring_max_size, sizeof(struct vocab_substring));
    vocab_hash = (int *) _mm_malloc(vocab_hash_size * sizeof(int), 64);
    substring_hash = (int *) _mm_malloc(vocab_hash_size * sizeof(int), 64);
    expTable = (real *) _mm_malloc((EXP_TABLE_SIZE + 1) * sizeof(real), 64);
    for (i = 0; i < EXP_TABLE_SIZE + 1; i++) {
        expTable[i] = exp((i / (real) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                    // Precompute f(x) = x / (x + 1)
    }

    printf("number of threads: %d\n", num_threads);
    printf("number of iterations: %d\n", iter);
    printf("hidden size: %d\n", hidden_size);
    printf("number of negative samples: %d\n", negative);
    printf("window size: %d\n", window);
    printf("batch size: %d\n", batch_size);
    printf("starting learning rate: %.5f\n", alpha);
    printf("stream from disk: %d\n", disk);
    printf("starting training using file: %s\n\n", train_file);

    if((model_type==0||model_type==1||model_type==2||model_type==3) && cbow_type == 1) {
        printf("model: CBOWNS\n");
        character_size = (MAX_CHINESE - MIN_CHINESE + 1);
        Train_CBOWBasedNS();
    }else if(model_type==0 && cbow_type == 0){
        printf("model: SGNS\n");
        Train_SGNS();
    }

    saveModel();
    return 0;
}
