$(document).ready(function() {
    // Remove old active link and update the current one
    $('li.active').removeClass('active');
    $('a[href="' + location.pathname + '"]').closest('li').addClass('active');
});