      $(document).ready(function(){
       document.getElementById("navForm").outerHTML=document.getElementById("navForm").outerHTML.replace('forecast', getURL());
       $("#dateRange").hide()
        graphs = ["MACD", "RSI", "graph", "returns", "ARIMA", "NN", "LSTM"]
        giveSwitchesOnclick(graphs);

        graphs.forEach(graph => {
            if ($("#" + graph + "checkbox").is(":checked") == false){
            $("#" + graph).hide();
            }
            else {
            $("#" + graph).show();
            }
        });
        function giveSwitchesOnclick(graphs){
            graphs.forEach(graph => {
                $("#" + graph + "checkbox").click(function () {
                    if ($("#" + graph + "checkbox").is(":checked") == false) {
                        $("#" + graph).hide();
                    }
                    else {
                        $("#" + graph).show();
                    }

                });
              });
            }
            });
    // Functions to replace the search button with relevent address for GET
    function getURL()
    {
        url = window.location.href;
        url = url.split('/')
        return url[url.length-1];
    }
