window.addEventListener('load', ()=> {
    let long, lat;
    let temperatureDescription = document.querySelector('.temperature-description');
    let temperatureDegree = document.querySelector('.temperature-degree');
    let locationTimezone = document.querySelector('.location-timezone');
    let temperatureSection = document.querySelector('.temperature');
    const temperatureSpan = document.querySelector('.temperature span');

    if(navigator.geolocation) 
    {
        navigator.geolocation.getCurrentPosition(
            position => {
                // console.log(position);
                long = position.coords.longitude;
                lat = position.coords.latitude;

                const proxy = `https://cors-anywhere.herokuapp.com/`;
                const api = `${proxy}https://api.darksky.net/forecast/3dd348194116fb1aea6305eb2cfd1633/${lat},${long}`;

                fetch(api)
                    .then(response =>{
                        return response.json();
                    })
                    .then(data =>{
                        // console.log(data);

                        // short hand of data.currently.temperature
                        const {temperature, summary, icon} = data.currently;

                        // Set DOM Elements from the API
                        temperatureDegree.textContent = temperature;
                        temperatureDescription.textContent = summary;
                        locationTimezone.textContent = data.timezone;
                        // Formula for Celsius
                        let celsius = (temperature - 32) * (5 / 9);

                        // Set Icon
                        setIcons(icon, document.querySelector('.icon'));

                        //Change temperature to Celsius/Fahrenheit
                        temperatureSection.addEventListener('click', () =>{
                            if(temperatureSpan.textContent === "F") {
                                temperatureSpan.textContent = "C";
                                temperatureDegree.textContent = (celsius);
                            } else {

                                temperatureSpan.textContent = "F";
                                temperatureDegree.textContent = temperature;
                            }
                        })

                    });     
        });

        
    } 
        
    else {
        h1.textContent = "Hey this is not working";
    }

    function setIcons(icon, iconID)
    {
        const skycons = new Skycons({color: "white"});
        const currentIcon = icon.replace(/-/g, "_").toUpperCase();
        skycons.play();
        return skycons.set(iconID, Skycons[currentIcon]);
    }

});