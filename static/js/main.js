/*
Copyright 2024 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

window.oncontextmenu = function() { return false; }

function insertUserPrompt() {
    prompt = $('#prompt').val()

    if (!prompt)
        return

    const subject = document.querySelector("#chat-history");
    
    injectHTML = `
    <div class="chat-message user">
      <div class="msg">`+ prompt + `</div>
    </div>`;

    subject.insertAdjacentHTML('beforeend', injectHTML);

    scrollToBottom();
}

function insertBotPlaceholder() {
    const subject = document.querySelector("#chat-history");
    
    injectHTML = `
    <div class="chat-message chatbot response-target">
        <div class="chat-loading-indicator-container">
            <div class="msg"><img id="loading-indicator" class="htmx-indicator" src="/static/images/loading.svg"/></div>
        </div>
    </div>`;

    subject.insertAdjacentHTML('beforeend', injectHTML);

    scrollToBottom();    
}

function scrollToBottom() {
    $('#chat-history')[0].scrollTop = $('#chat-history')[0].scrollHeight;
}

function enableFormFields() {
    $("#chat-form > input[type='text'], #chat-form > button").attr('disabled', false);    
    $('#prompt').focus();
}

function disableFormFields() {
    $("#chat-form > input[type='text'], #chat-form > button").attr('disabled', true);    
}

function showLoadingIndicator() {
    $("#loading-indicator").css('opacity', '100%');
}

function removeLoadingIndicator() {
    $('.chat-loading-indicator-container').remove();    
    $('.response-target').removeClass('response-target');
}

// Audio helpers

let mediaRecorder;
let audioChunks = [];

async function startRecording(self, event) {
    if ("buttons" in event) {
        if (event.buttons == 2)
            return;
    }    
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.start();
    mediaRecorder.ondataavailable = event => audioChunks.push(event.data);

    insertUserPrompt("&#127911;");
}

function stopRecording(self, event) {
    mediaRecorder.stop();

    mediaRecorder.onstop = () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        audioChunks = []; // Reset for the next recording

        const formData = new FormData();
        formData.append('audio', blob, 'recording.webm');

        disableFormFields();
        disableRecordButton();
        insertBotPlaceholder();
        showLoadingIndicator();

        fetch('/chat', { method: 'POST', body: formData })
            .then(response => response.text()).then((dataStr) => { postRecodingCallback(dataStr); scrollToBottom(); removeLoadingIndicator(); })
            .catch(error => console.error('Error sending audio message:', error));
    };
}

function postRecodingCallback(response) {
    const divFragment = document.createRange().createContextualFragment(response);
    const subject = document.querySelector(".response-target");
    subject.append(divFragment);

    enableFormFields(false);
    enableRecordButton();
}

function enableRecordButton() {
    $("#talk-button").attr('disabled', false);
}

function disableRecordButton() {
    $("#talk-button").attr('disabled', true);    
}
