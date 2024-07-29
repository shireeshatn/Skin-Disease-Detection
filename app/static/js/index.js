// static/js/index.js

$(document).ready(function () {
    var is_detect = true

    $('#detect-condition').click(function () {
        is_detect = true

        $('#thumbnails').fadeOut(500, function () {
            $('#backButton').fadeIn(500).attr('hidden', false)
            $('#upload-section').fadeIn(500).attr('hidden', false);
        });
    });

    $('#track-condition').click(function () {
        is_detect = false

        $.ajax({
            url: '/check_login',  // Endpoint to check login status
            method: 'GET',
            success: function (response) {
                if (response.logged_in) {
                    $('#thumbnails').fadeOut(500, function () {
                        $('#backButton').fadeIn(500).attr('hidden', false)
                        $('#upload-section').fadeIn(500).attr('hidden', false);
                    });
                } else {
                    window.location.href = '/login';
                }
            }
        });
    });

    function hideUploadSection() {
        $('#upload-section').fadeOut(500, function () {
            $('#backButton').fadeOut(500)
            $('#thumbnails').fadeIn(500);
        });
    }

    $('#backButton').click(function () {
        hideUploadSection();
    });

    $('#upload-form').submit(function (event) {
        event.preventDefault();
        var formData = new FormData(this);
        var detect_values = {
            url: '/detect_disease',
            method: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                $('#loading-spinner').attr('hidden', true);
                if (response['error']) {
                    $('#result').html('<p style="color: red;">Error: ' + response['error'] + '</p>').fadeIn(500).attr('hidden', false);
                } else {
                    const diseases = response['disease_names'].split(',')
                    const predictions = response['probabilities'].split(',')
                    var messageHtml = ''

                    for (let i = 0; i < 5; i++) {
                        messageHtml += '<p><a href=https://dermnetnz.org/topics/' + diseases[i]
                            + ' class=\"my-link\">' + diseases[i] + '</a>: ' + predictions[i] + '<p>';
                    }

                    $('#result').html(messageHtml).fadeIn(500).attr('hidden', false);
                }

                $('#upload-section button').fadeOut(500);
            },
            error: function (xhr, status, error) {
                $('#loading-spinner').attr('hidden', true);
                console.error('Error:', error);
                $('#result').html('<p style="color: red;">Error: ' + error + '</p>').fadeIn(500).attr('hidden', false);
            },
        };
        var track_values = {
            url: '/track_acne',
            method: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function (response) {
                if (response['error']) {
                    $('#result').html('<p style="color: red;">Error: ' + response['error'] + '</p>').fadeIn(500).attr('hidden', false);
                } else {
                    // Example: Display response['message'] as a JSON table
                    const level = response['current_level']
                    var messageHtml = '<p> Predicted Acne Level:   ' + level + '<p>';
                    $('#result').html(messageHtml).fadeIn(500).attr('hidden', false);
                }

                $('#upload-section button').fadeOut(500);
            },
            error: function (xhr, status, error) {
                $('#loading-spinner').attr('hidden', true);
                console.error('Error:', error);
                $('#result').html('<p style="color: red;">Error: ' + error + '</p>').fadeIn(500).attr('hidden', false);
            },
        };
        $.ajax(is_detect ? detect_values : track_values);
    });
});
