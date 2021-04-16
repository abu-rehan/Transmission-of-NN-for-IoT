/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>
#include <ArduinoBLE.h>
#include "main_functions.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

int g_model_len=0;
unsigned char* g_model= nullptr;

const bool modelreceived= false;

BLEService modelService("BE050000-117D-453B-A18D-C76455B19B36");
// BLE Characteristics - custom 128-bit UUIDs, read and writable by central
BLECharacteristic modelSizeCharacteristic("BE050001-117D-453B-A18D-C76455B19B36", BLERead | BLEWrite, 4);
BLECharacteristic modelByteCharacteristic("BE050002-117D-453B-A18D-C76455B19B36", BLERead | BLEWrite, 160);

int byte_array_to_int(unsigned char* b_arr);
void setupSerial();
void setupBLE();
void receiveModel();
void initializeInterpreter();

// The name of this function is important for Arduino compatibility.
void setup() {
  setupSerial();
  setupBLE();
  receiveModel();
  initializeInterpreter();
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Calculate an x value to feed into the model. We compare the current
  // inference_count to the number of inferences per cycle to determine
  // our position within the range of possible x values the model was
  // trained on, and use this to calculate a value.
  float position = static_cast<float>(inference_count) /
                   static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  // Quantize the input from floating-point to integer
  int8_t x_quantized = x / input->params.scale + input->params.zero_point;
  // Place the quantized input in the model's input tensor
  input->data.int8[0] = x_quantized;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n",
                         static_cast<double>(x));
    return;
  }

  // Obtain the quantized output from model's output tensor
  int8_t y_quantized = output->data.int8[0];
  // Dequantize the output from integer to floating-point
  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x, y);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}

int byte_array_to_int(unsigned char* b_arr){
  int* numb= (int*) b_arr;
  return *numb;
}

void setupSerial(){
  Serial.begin(9600);
  while(!Serial);
}

void setupBLE()
{
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);
  }
  // set advertised local name and service UUID:
  BLE.setLocalName("NeuralNetIoT");
  BLE.setAdvertisedService(modelService);

  // add the characteristics to the service
  modelService.addCharacteristic(modelSizeCharacteristic);
  modelService.addCharacteristic(modelByteCharacteristic);

  // add service
  BLE.addService(modelService);

  BLE.advertise();

  Serial.println("BLE is ready");
}

void receiveModel(){
    int no_of_batches, stage=1, bc=0;
    Serial.println("Ready to receive the model");
    while(!modelreceived){
      // listen for BLE peripherals to connect:
      BLEDevice central = BLE.central();
      // if a central is connected to peripheral:
      if (central) {
        Serial.print("Connected to central: ");
        // print the central's MAC address:
        Serial.println(central.address());
        // while the central is still connected to peripheral:
        while (central.connected()) {
              if(modelSizeCharacteristic.written() && stage==1){
                g_model_len= byte_array_to_int((unsigned char *)modelSizeCharacteristic.value());
                g_model= (unsigned char*) malloc(g_model_len);
                if(g_model==NULL || g_model==nullptr){
                  Serial.println("Allocation failed");
                  while(1);
                }
                no_of_batches=(g_model_len/160);
                Serial.print("Model size: ");
                Serial.println(g_model_len);
                stage=2;
              }
              if (modelByteCharacteristic.written() && stage==2){
                    memcpy((g_model+bc*160),(unsigned char *)modelByteCharacteristic.value(), 160);
                    bc++;
                    Serial.print("Received batch: ");
                    Serial.println(bc);
                    if(bc>=no_of_batches)
                      break;
              }
        }
        break;
      }
    }
}

void initializeInterpreter()
{
  Serial.println("Initializing interpreter...");
  delay(3000);
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}
