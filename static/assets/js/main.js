/*
	Prologue by HTML5 UP
	html5up.net | @ajlkn
	Free for personal and commercial use under the CCA 3.0 license (html5up.net/license)
*/


(function($) {




    var $window = $(window),
        $body = $('body'),
        $nav = $('#nav');
    var select = 0;
    // Breakpoints.
    /*
    breakpoints({
        wide: ['961px', '1880px'],
        normal: ['961px', '1620px'],
        narrow: ['961px', '1320px'],
        narrower: ['737px', '960px'],
        mobile: [null, '736px']
    });

    */

    // Play initial animations on page load.
    $window.on('load', function() {
        window.setTimeout(function() {
            $body.removeClass('is-preload');
        }, 100);
    });

    // Nav.
    var $nav_a = $nav.find('a');

    $nav_a
        .addClass('scrolly')
        .on('click', function(e) {

            var $this = $(this);

            // External link? Bail.
            if ($this.attr('href').charAt(0) != '#')
                return;

            // Prevent default.
            e.preventDefault();

            // Deactivate all links.
            $nav_a.removeClass('active');

            // Activate link *and* lock it (so Scrollex doesn't try to activate other links as we're scrolling to this one's section).
            $this
                .addClass('active')
                .addClass('active-locked');

        })
        .each(function() {

            var $this = $(this),
                id = $this.attr('href'),
                $section = $(id);

            // No section for this link? Bail.
            if ($section.length < 1)
                return;

            // Scrollex.
            $section.scrollex({
                mode: 'middle',
                top: '-10vh',
                bottom: '-10vh',
                initialize: function() {

                    // Deactivate section.
                    $section.addClass('inactive');
                },
                enter: function() {

                    // Activate section.
                    $section.removeClass('inactive');


                    // No locked links? Deactivate all links and activate this section's one.
                    if ($nav_a.filter('.active-locked').length == 0) {

                        $nav_a.removeClass('active');
                        $this.addClass('active');

                    }

                    // Otherwise, if this section's link is the one that's locked, unlock it.
                    else if ($this.hasClass('active-locked'))
                        $this.removeClass('active-locked');

                }
            });

        });

    // Delete

    $('.delete').click(function() {

        var check = confirm("정말 삭제하시겠습니까?");

        if (check == true && select == 1) {
            alert("삭제되었습니다.");
        } else if (check == true && select == 0) {
            alert("Error : 삭제할 목록을 선택하지 않았습니다.");
        }

        select = 0;

    })



    $('.list').click(function() {
        select = 1;
    })



    //Start
    $('#main-screen').hide();

    $('.streaming-start').click(function() {
        $('#main-screen').show();
    });

    //Pause
    $('.streaming-pause').click(function() {
        $('#main-screen').hide();
    });

    //Stop
    $('.streaming-stop').click(function() {
        $('#main-screen').hide();
    });

    //Spinner
    $('#loading').hide();
    $('.okay').hide();

    $('.shutter').click(function() {
        $('#loading').show();
        setTimeout(() => {
            $('.shutter').hide();
            $('#loading').hide();
            $('.okay').show();
        }, 3000);
    })



    //Add

    $('.add').click(function() {
        var $camera = $(this).data('id');
        layer_popup($camera);
    });

    //camera

    function layer_popup(el) {
        $('#' + el).fadeIn();
        $('#' + el).find('a.layer-exit').click(function() {
            $('#' + el).fadeOut();
            return false;
        });

        $('.layer .dimBg').click(function() {
            $('.dim-layer').fadeOut();
            return false;
        });

        $('.okay').click(function() {
            $('#' + el).fadeOut();
            $('.dim-layer').fadeOut();
            $('.okay').hide();
            $('.shutter').show();
        });

    }





    // Scrolly.
    $('.scrolly').scrolly();

    // Header (narrower + mobile).

    // Toggle.
    $(
            '<div id="headerToggle">' +
            '<a href="#header" class="toggle"></a>' +
            '</div>'
        )
        .appendTo($body);

    // Header.
    $('#header')
        .panel({
            delay: 500,
            hideOnClick: true,
            hideOnSwipe: true,
            resetScroll: true,
            resetForms: true,
            side: 'left',
            target: $body,
            visibleClass: 'header-visible'
        });



})(jQuery);